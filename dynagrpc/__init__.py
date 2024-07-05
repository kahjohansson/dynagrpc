"""DynagRPC Python abstraction library over gRPC and protobuf types."""
from __future__ import annotations

from functools import partial
from itertools import groupby
from operator import attrgetter
import re
import warnings

from google.protobuf.descriptor import (
    Descriptor,
    EnumDescriptor,
    FieldDescriptor,
)
from google.protobuf.message import Message

from . import _astcast

try:
    from typing import Literal
except ImportError:  # TODO: Drop Python 3.7 compatibility
    from typing_extensions import Literal


__version__ = "0.1.0.dev"

__all__ = [
    "DynaGrpcWarning",
    "AttrDict",
    "snake2pascal",
    "pascal2snake",
    "create_enum_dict",
    "GrpcTypeCastRegistry",
]


class DynaGrpcWarning(Warning):
    pass


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def snake2pascal(name: str) -> str:
    """Convert snake_case to PascalCase (a.k.a. UpperCamelCase)."""
    return "".join(map(str.title, name.split("_")))


def pascal2snake(name: str) -> str:
    """Convert PascalCase to snake_case."""
    return "_".join(
        ("" if length == 1 else "_").join(group).lower()
        for length, group in groupby(re.findall("[A-Z][^A-Z]*", name), len)
    )


def create_enum_dict(
    enum_type: EnumDescriptor,
    mode: Literal["int2str", "str2int"],
    prefix: str | None = None,
) -> dict[int, str] | dict[str, int]:
    """
    Dictionary representing a protobuf enum assuming no alias,
    already removing the common prefix if it's following the convention
    of using the enum name in ``UPPER_CASE_`` as the prefix.

    Parameters
    ----------
    enum_type :
        Enum type from the ``service_pb2`` (or ``GrpcServer._protos``)
        to be converted to a mapping.
    mode :
        Whether the result should cast integer enum codes to strings
        (``int2str``) or vice-versa (``str2int``).
    prefix :
        Custom prefix to be removed, use an empty string to force it to
        don't cut prefixes; ``None`` (default) means it should attempt
        to use the upper ``snake_case`` of the enum type name with a
        trailing underscore (``_``).
    """
    common = prefix
    if prefix is None:
        common = pascal2snake(enum_type.name).upper() + "_"
    else:
        common = prefix
    if all(value.name.startswith(common) for value in enum_type.values):
        threshold = len(common)
    elif prefix is not None:
        raise ValueError(f"The {prefix!r} prefix is not common for all values")
    else:
        warnings.warn(
            f"Missing values prefix in enum {enum_type.full_name}",
            DynaGrpcWarning,
        )
        threshold = 0
    if mode == "int2str":
        pairs = enum_type.values_by_number.items()
        return {number: value.name[threshold:] for number, value in pairs}
    if mode == "str2int":
        pairs = enum_type.values_by_name.items()
        return {name[threshold:]: value.number for name, value in pairs}
    raise ValueError(f"Unknown mode {mode!r}")


class GrpcTypeCastRegistry:
    """
    Registry of callables and dictionaries intended for representing
    and converting enums, as well as for type casting between Python
    objects and gRPC-specific protobuf messages or fields.
    """

    def __init__(self, dict_cls: type[dict] = AttrDict):
        # Registry names expected by the AST-generated lambda functions
        # Keys for the registries are always the "descriptor" full name
        self._namespace = AttrDict(
            d=dict_cls,  # Dict wrapper for all messages
            m2p={},  # Message to Python (usually dict) registry
            p2m={},  # Python to protobuf message registry
            f2p={},  # Field to Python registry
            p2f={},  # Python to single protobuf message field registry
            i2s={},  # Enum registry, int to str
            s2i={},  # Enum registry, str to int
            c={},  # Message constructors registry
        )  # Note: _namespace is not intended to be accessed directly!
        self._registering = set()
        self._register_google_types()

    def _register_google_types(self) -> None:
        """
        Register ``google.protobuf.*`` types that behave differently
        than custom protobuf-defined types.
        """
        # Register the NullValue enum from struct.proto
        self._namespace.i2s["google.protobuf.NullValue"] = {0: None}
        self._namespace.s2i["google.protobuf.NullValue"] = {None: 0}

        # Register all wrapped scalars from wrappers.proto
        for full_name, wrapper in _astcast.GOOGLE_PROTOBUF_WRAPPERS.items():
            self._namespace.c[full_name] = wrapper
            self._namespace.m2p[full_name] = attrgetter("value")
            self._namespace.p2m[full_name] = _astcast.create_lambda(
                return_ast=_astcast.wrapper2message_ast(
                    full_name=full_name,
                    value_name="arg",
                    constructors_regname="c",
                ),
                arg_name="arg",
                file_name=f"<p2m/{full_name}>",
                namespace=self._namespace,
            )

        # Register a failure for these not yet implemented types
        def fail(error_message, unused_input):
            raise NotImplementedError(error_message)

        for name in (
            "ListValue", "Struct", "Value",  # struct.proto
            "Any",  # any.proto
            "Duration",  # duration.proto
            "FieldMask",  # field_mask.proto
            "Timestamp",  # timestamp.proto
        ):
            full_name = "google.protobuf." + name
            named_fail = partial(fail, full_name)
            self._namespace.m2p[full_name] = named_fail
            self._namespace.p2m[full_name] = named_fail

    def register_enum_type(self, enum_type: EnumDescriptor) -> None:
        full_name = enum_type.full_name
        self._namespace.i2s[full_name] = create_enum_dict(enum_type, "int2str")
        self._namespace.s2i[full_name] = create_enum_dict(enum_type, "str2int")

    def register_message_type(self, message_type: Descriptor) -> None:
        full_name = message_type.full_name

        # Prevent overwriting messages (e.g. google wrappers)
        if full_name in self._namespace.m2p:
            return  # Nothing to do

        self._registering.add(message_type)  # Prevent reentrant deadlock
        try:
            # Create a lambda to cast a protobuf message to a custom dict
            self._namespace.m2p[full_name] = _astcast.create_lambda(
                return_ast=_astcast.wrap_call_ast(
                    callable_name="d",
                    input_ast=_astcast.message2dict_ast(
                        message_type=message_type,
                        value_name="msg",
                        enum_registry=self._namespace.i2s,
                        field_regname="f2p",
                    ),
                ),
                arg_name="msg",
                file_name=f"<m2p/{full_name}>",
                namespace=self._namespace,
            )

            # Create a lambda to cast a dict to protobuf message
            self._namespace.p2m[full_name] = _astcast.create_lambda(
                return_ast=_astcast.dict2message_ast(
                    message_type=message_type,
                    value_name="data",
                    constructors_regname="c",
                    field_regname="p2f",
                ),
                arg_name="data",
                file_name=f"<p2m/{full_name}>",
                namespace=self._namespace,
            )

            # Populate self._namespace.f2p and self._namespace.p2f
            for field in message_type.fields:
                self.register_field_type(field)

            # Unfortunately, the constructor of the message is private,
            # but accessing it is more straightforward than looking for
            # its underlying Python module
            self._namespace.c[full_name] = message_type._concrete_class
        finally:
            self._registering.remove(message_type)

    def register_field_type(self, field: FieldDescriptor) -> None:
        # Ensure all message types are registered
        mt = field.message_type
        if mt and not (mt.GetOptions().map_entry or mt in self._registering):
            self.register_message_type(mt)

        # Create lambdas to cast the field to/from Python
        for registry, msg_regname, enum_regname in (
            (self._namespace.f2p, "m2p", "i2s"),
            (self._namespace.p2f, "p2m", "s2i"),
        ):
            registry[field.full_name] = _astcast.create_lambda(
                return_ast=_astcast.field_cast_ast(
                    msg_regname=msg_regname,
                    enum_regname=enum_regname,
                    field=field,
                    value_name="value",
                ),
                arg_name="value",
                file_name=f"<{msg_regname}/{field.full_name}>",
                namespace=self._namespace,
            )

    def proto2py(
        self,
        message: Message,
    ) -> dict | str | int | bool | float | bytes:
        """
        Convert a gRPC-specific protobuf message to a Python object,
        generally an instance of the given ``dict_cls`` (a dictionary).

        Though not based on the proto3 JSON spec, this converter is an
        alternative to ``google.protobuf.json_format.MessageToDict``.
        Some important similarities and differences that should be
        highlighted:

        - Result is a dictionary, not a custom object, unless the
          message type is a special ``google.protobuf.*`` one.
        - Items in the result keep the protobuf message number order.
        - The resulting dictionary has attribute access to its items,
          as long as it doesn't clash with dictionary methods.
        - It always includes all fields in the resulting dictionary,
          using ``None`` as the  placeholder value for ``optional``
          fields; all protobuf message types are implicitly optional.
        - Use the default value where applicable, similar to
          ``MessageToDict(..., including_default_value_fields=True)``
          when the field should not be ``None``.
        - Keys are original field names from protobuf definition, like
          ``MessageToDict(..., preserving_proto_field_name=True)``
        - Enum values are shortened strings, unlike any
          ``MessageToDict`` capability.
        - Mapping key type is kept as defined in the proto file, like
          ``rpc.json_format.PreserveIntMessageToDict`` for ``int`` keys
          and unlike any JSON-based alternative for other key types
          like ``bool`` or ``bytes``.
        """
        return self._namespace.m2p[message.DESCRIPTOR.full_name](message)

    def py2proto(self, message_cls: type[Message], data: dict) -> Message:
        """Convert a dictionary to a gRPC-specific protobuf message."""
        return self._namespace.p2m[message_cls.DESCRIPTOR.full_name](data)
