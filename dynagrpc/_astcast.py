"""
JIT AST generation and compiling for type casters of dictionaries and
other Python types from/to protobuf messages, fields and enums.

This module is intended to be used only internally by dynagrpc.
"""
from __future__ import annotations

import ast
from collections.abc import Callable
from functools import partial
from operator import attrgetter
from typing import Any

from google.protobuf import wrappers_pb2
from google.protobuf.descriptor import (
    Descriptor,
    EnumDescriptor,
    FieldDescriptor,
)


def create_lambda(
    return_ast: ast.AST,
    arg_name: str = "value",
    file_name: str = "<unknown>",
    namespace: dict[str, Any] | None = None,
    optimize: int = 2,
) -> Callable:
    """
    Create a single-argument lambda function with the chosen argument
    name and whose returning body is the given AST, exposing to its
    lexical scope only the given namespace.
    """
    # Since ast.Lambda changed in Python 3.8, use what's available
    func_ast = ast.parse("lambda tmp: tmp", mode="eval")
    func_ast.body.args.args[0].arg = arg_name
    func_ast.body.body = return_ast
    return eval(
        compile(  # JIT creating the lambda function
            source=ast.fix_missing_locations(func_ast),
            filename=file_name,
            mode="eval",
            optimize=optimize,
        ),
        namespace,
    )


def get_field_descriptor_map(prefix: str) -> dict[int, str]:
    """Collect ``FieldDescriptor`` attributes with the given prefix."""
    size = len(prefix)
    keys = (key for key in dir(FieldDescriptor) if key.startswith(prefix))
    pairs = ((getattr(FieldDescriptor, key), key[size:]) for key in keys)
    return dict(sorted(pairs))


CPPTYPE_MAP = get_field_descriptor_map("CPPTYPE_")
LABEL_MAP = get_field_descriptor_map("LABEL_")
TYPE_MAP = get_field_descriptor_map("TYPE_")  # From google.protobuf.Field.Kind
CPPTYPE_SCALARS = (  # Scalars are already Python objects (even if repeated)
    "FLOAT", "DOUBLE",  # Stored as float
    "INT32", "INT64", "UINT32", "UINT64",  # Stored as int
    "STRING",  # Stored as either str or bytes (distinct in _TYPE_MAP)
    "BOOL",  # Stored as bool (True or False)
)


def enum_cast_ast(
    registry_name: str,
    enum_type: EnumDescriptor,
    value_name: str,
) -> ast.AST:
    """
    AST for ``registry[enum_name].get(value, value)``, assuming the
    registry is a dictionary of dictionaries representing enums.
    """
    full_name = enum_type.full_name
    source = f"{registry_name}[{full_name!r}].get({value_name}, {value_name})"
    return ast.parse(source, mode="eval").body


def message_cast_ast(
    registry_name: str,
    message_type: Descriptor,
    value_name: str,
) -> ast.AST:
    """
    AST for ``registry[message_name](value)``, assuming the registry is
    a dictionary of message converters (callables).
    """
    python_code = f"{registry_name}[{message_type.full_name!r}]({value_name})"
    return ast.parse(python_code, mode="eval").body


def label_caster_wrap_ast(
    label: str,
    caster: Callable,
    value_name: str,
):
    if label == "REPEATED":
        return ast.ListComp(  # [caster(v) for v in value]
            elt=caster("v"),
            generators=[ast.comprehension(
                ast.Name("v", ctx=ast.Store()),
                ast.Name(value_name, ctx=ast.Load()),
                ifs=[],
                is_async=0,
            )],
        )
    return caster(value_name)


def field_cast_ast(
    msg_regname: str,
    enum_regname: str,
    field: FieldDescriptor,
    value_name: str,
) -> ast.AST:
    """
    AST to cast a ``value`` between protobuf and Python representations
    of the given fields, assuming two dicts for registry are available
    in lexical scope, whose names are given as parameters.
    """
    ctype = CPPTYPE_MAP[field.cpp_type]
    if ctype in CPPTYPE_SCALARS:  # Python objects even when repeated
        return ast.Name(value_name, ctx=ast.Load())

    if ctype == "ENUM":
        return label_caster_wrap_ast(
            label=LABEL_MAP[field.label],
            caster=partial(enum_cast_ast, enum_regname, field.enum_type),
            value_name=value_name,
        )

    if ctype == "MESSAGE":  # Messages, repeated messages and maps

        # Handle map<key, value> avoiding spurious calls / comprehensions
        if field.message_type.GetOptions().map_entry:
            key_type = field.message_type.fields_by_name["key"]
            key_ctype = CPPTYPE_MAP[key_type.cpp_type]
            if key_ctype not in CPPTYPE_SCALARS:
                raise RuntimeError(f"Invalid key C++ type {key_ctype}")
            value_type = field.message_type.fields_by_name["value"]
            value_ctype = CPPTYPE_MAP[value_type.cpp_type]
            if value_ctype in CPPTYPE_SCALARS:  # Nothing to do
                return ast.Name(value_name, ctx=ast.Load())
            dict_comp_ast = ast.parse(
                "{k: nest_tmp for k, v in value_tmp.items()}",
                mode="eval",
            ).body  # *_tmp are replaced below, in order
            dict_comp_ast.value = field_cast_ast(
                msg_regname=msg_regname,
                enum_regname=enum_regname,
                field=value_type,
                value_name="v",
            )
            dict_comp_ast.generators[0].iter.func.value.id = value_name
            return dict_comp_ast

        return label_caster_wrap_ast(
            label=LABEL_MAP[field.label],
            caster=partial(message_cast_ast, msg_regname, field.message_type),
            value_name=value_name,
        )

    raise RuntimeError(f"Unknown field C++ type {ctype}")


def field_default_ast(
    field: FieldDescriptor,
    enum_registry: dict[str, dict[int, str | None]],
) -> ast.AST:
    """AST for the default field value."""
    ctype = CPPTYPE_MAP[field.cpp_type]
    if ctype == "MESSAGE" and field.message_type.GetOptions().map_entry:
        return ast.Dict(keys=[], values=[])
    if LABEL_MAP[field.label] == "REPEATED":
        return ast.List(elts=[], ctx=ast.Load())
    if field.has_presence:  # Includes all non-map messages (implicitly)
        return ast.Constant(None)
    if ctype == "ENUM":
        return ast.Constant(enum_registry[field.enum_type.full_name][0])
    if ctype == "BOOL":
        return ast.Constant(False)
    if ctype == "STRING":
        if TYPE_MAP[field.type] == "BYTES":
            return ast.Constant(b"")
        return ast.Constant("")
    if ctype in ("FLOAT", "DOUBLE"):
        return ast.Constant(0.)
    if ctype in ("INT32", "INT64", "UINT32", "UINT64"):
        return ast.Constant(0)
    raise RuntimeError(f"Unknown field C++ type {ctype}")


def message2dict_ast(
    message_type: Descriptor,
    value_name: str,
    enum_registry: dict[str, dict[int, str | None]],
    field_regname: str,
) -> ast.AST:
    """
    AST to cast a ``value`` message to a Python dictionary.

    Parameters
    ----------
    message_type :
        Concrete ``value.DESCRIPTOR`` object.
    value_name :
        Name of the ``value`` variable as seen in AST lexical scope.
    enum_registry :
        Registry of enums represented as "int to str" dictionaries.
    field_regname :
        Name of the registry of "field to Python" converters as seen
        in the AST lexical scope.
    """
    fields = sorted(message_type.fields, key=attrgetter("number"))
    defaults = [field_default_ast(field, enum_registry) for field in fields]

    # Create a dict AST node resembling
    #   {f0: f0_default, f1: f1_default, ..., **dict_comp}
    # That is, a constant dictionary with each field ordered
    # and mapped to its respective default value, and expanded
    # at the end to include all filled message values
    dict_comp_ast = ast.parse(
        "{f.name: regname[f.full_name](v) for f, v in tmp.ListFields()}",
        mode="eval",
    ).body
    dict_comp_ast.value.func.value.id = field_regname
    dict_comp_ast.generators[0].iter.func.value.id = value_name
    return ast.Dict(
        keys=[ast.Constant(field.name) for field in fields] + [None],
        values=defaults + [dict_comp_ast],
    )


def dict2message_ast(
    message_type: Descriptor,
    value_name: str,
    constructors_regname: str,
    field_regname: str,
) -> ast.AST:
    """
    AST to cast a ``value`` Python dictionary to a protobuf message.

    Parameters
    ----------
    message_type :
        Concrete ``value.DESCRIPTOR`` object.
    value_name :
        Name of the ``value`` variable as seen in AST lexical scope.
    constructors_regname :
        Name of the registry of protobuf message constructors as seen
        in the AST lexical scope.
    field_regname :
        Name of the registry of "Python object to protobuf field"
        converters as seen in the AST lexical scope.
    """
    # In the simplest form, it's just a constructors[message_name](**value)
    message_name = repr(message_type.full_name)
    python_code = f"{constructors_regname}[{message_name}](**{value_name})"
    result_ast = ast.parse(python_code, mode="eval").body

    # If a field needs conversion, all inputs need to be checked, and
    # the **value keyword arguments should be a dict comprehension
    if any(map(is_nesting_field, message_type.fields)):
        dict_comp_ast = ast.parse(
            "{k: nest_tmp for k, v in value_tmp.items()}",
            mode="eval",
        ).body  # *_tmp are replaced below, in order
        dict_comp_ast.value = kv_call_ast(
            message_type=message_type,
            key_name="k",
            value_name="v",
            registry_name=field_regname,
        )
        dict_comp_ast.generators[0].iter.func.value.id = value_name
        result_ast.keywords[0].value = dict_comp_ast

    return result_ast


def is_nesting_field(field: FieldDescriptor) -> bool:
    """
    Check if a conversion callable is required to wrap this field when
    constructing its underlying message.
    """
    ctype = CPPTYPE_MAP[field.cpp_type]
    if ctype == "ENUM":
        return True  # To cast to int
    if ctype == "MESSAGE":
        message_type = field.message_type
        if message_type.GetOptions().map_entry:
            return is_nesting_field(message_type.fields_by_name["value"])
        return True
    return False


def kv_call_ast(
    message_type: Descriptor,
    key_name: str,
    value_name: str,
    registry_name: str,
) -> ast.AST:
    """
    AST for ``registry[prefix + key](value)`` where ``prefix`` is the
    full message name with a trailing dot.
    """
    prefix = message_type.full_name + "."
    python_code = f"{registry_name}[{prefix!r} + {key_name}]({value_name})"
    return ast.parse(python_code, mode="eval").body


def wrap_call_ast(callable_name: str, input_ast: ast.AST) -> ast.AST:
    return ast.Call(
        func=ast.Name(callable_name, ctx=ast.Load()),
        args=[input_ast],
        keywords=[],
    )


GOOGLE_PROTOBUF_WRAPPERS = {  # Message constructors registry for wrappers
    f"google.protobuf.{pascal_name}": getattr(wrappers_pb2, pascal_name)
    for upper_prefix in CPPTYPE_SCALARS + ("BYTES",)
    for pascal_name in [upper_prefix.title().replace("int", "Int") + "Value"]
}


def wrapper2message_ast(
    full_name: str,
    value_name: str,
    constructors_regname: str,
) -> ast.AST:
    """
    AST to cast ``value`` to a google protobuf wrapper message.

    Parameters
    ----------
    full_name :
        Full wrapper name, like ``google.protobuf.BoolValue``.
    value_name :
        Name of the ``value`` variable as seen in AST lexical scope.
    constructors_regname :
        Name of the registry of protobuf message constructors as seen
        in the AST lexical scope.
    """
    python_code = f"{constructors_regname}[{full_name!r}](value={value_name})"
    return ast.parse(python_code, mode="eval").body
