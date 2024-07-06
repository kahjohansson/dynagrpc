"""DynagRPC Python abstraction library over gRPC and protobuf types."""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from importlib import import_module
from inspect import signature
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
import grpc

from . import _astcast

try:
    from typing import Literal
except ImportError:  # TODO: Drop Python 3.7 compatibility
    from typing_extensions import Literal


__version__ = "0.1.0"

__all__ = [
    "DynaGrpcError",
    "ServiceNotFound",
    "TooManyServices",
    "UnknownServiceName",
    "SignatureMismatch",
    "AlreadyRegisteredHandler",
    "AlreadyRegisteredName",
    "DynaGrpcWarning",
    "AttrDict",
    "snake2pascal",
    "pascal2snake",
    "create_enum_dict",
    "GrpcTypeCastRegistry",
    "GrpcServiceBase",
    "GrpcServer",
    "GrpcTestClient",
]


class DynaGrpcError(Exception):
    """Base for dynagrpc errors, not intended to be raised directly."""


class ServiceNotFound(DynaGrpcError):
    pass


class TooManyServices(DynaGrpcError):
    pass


class UnknownServiceName(DynaGrpcError):
    pass


class SignatureMismatch(DynaGrpcError):
    pass


class AlreadyRegisteredHandler(DynaGrpcError):
    pass


class AlreadyRegisteredName(DynaGrpcError):
    pass


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


class GrpcServiceBase:
    """
    Abstraction over ``service_pb2`` and ``service_pb2_grpc``
    for accessing a single service from a protobuf file.
    """

    # TODO: Review this signature, what would change if the protobuf
    # definition is generated instead?
    def __init__(
        self,
        pkg_name: str,
        service_name: str | None = None,
        module_name: str = "service",
        *,
        import_with: Literal["importlib", "grpc"] = "grpc",
    ):
        """
        Parameters
        ----------
        pkg_name :
            Python package where the gRPC service was generated,
            e.g. for a ``somewhere/example/service.proto``
            the package would be ``somewhere.example``.
        service_name :
            Service name as defined in ``service.proto`` (source file),
            defaults to the first service name found in it.
        module_name :
            Stem of the protobuf file, e.g. for ``service.proto`` it's
            just ``service``.
        import_with :
            String to choose between ``importlib`` or ``grpc``
            as the importing mechanism to be used.
            The ``grpc`` choice requires he ``grpcio-tools`` library,
            whereas the ``importlib`` choice requires generated Python
            files from the protobuf definition.
        """
        self.pkg_name = pkg_name
        self.typecast = GrpcTypeCastRegistry()

        # Import the modules and store its related attributes
        pkg_name_slash = pkg_name.replace(".", "/")
        self._source = source = f"{pkg_name_slash}/{module_name}.proto"
        if import_with == "grpc":
            source = self._source
            self._protos, self._services = grpc.protos_and_services(source)
        elif import_with == "importlib":
            importer = partial(import_module, package=pkg_name)
            self._services = importer(f".{module_name}_pb2_grpc")
            self._protos = importer(f".{module_name}_pb2")
        else:
            raise ValueError("Invalid import_with")
        self._imported_with = import_with

        # Find the single-service name from the service_pb2 module
        all_services = self._protos.DESCRIPTOR.services_by_name
        if not all_services:
            raise ServiceNotFound("No service in proto file")
        if not service_name:
            if len(all_services) > 1:
                raise TooManyServices(str(sorted(all_services)))
            service_name, = all_services  # Comma = get the only key
        elif service_name not in all_services:
            raise UnknownServiceName(f"Name {service_name} isn't a service")
        self._name = service_name

    def register_types_from_proto(self):
        protos_descriptor = self._protos.DESCRIPTOR
        for enum_type in protos_descriptor.enum_types_by_name.values():
            self.typecast.register_enum_type(enum_type)
        for message_type in protos_descriptor.message_types_by_name.values():
            self.typecast.register_message_type(message_type)


class GrpcServer(GrpcServiceBase):
    """
    Implementation of a gRPC server and RPC handler registry.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = {}
        self.name_map = {}  # From snake_case_handler to RpcCommandHandler
        self.exception_handlers = {}

        # Get binding information & init a servicer-inherited class
        servicer_name = self._name + "Servicer"
        self.servicer = getattr(self._services, servicer_name)
        self.binder = getattr(self._services, f"add_{servicer_name}_to_server")
        self._implementation = type(self._name, (self.servicer,), {})()

        # Get all enums and messages
        self.register_types_from_proto()

    def rpc(
        self,
        name: str | None = None,
        *,
        request_name: str | None = None,
        response_name: str | None = None,
        output: Literal["auto", "single", "dict", "tuple", "bypass"] = "auto",
        cast: dict[str, Callable] | None = None,
    ):
        """
        Parametrized decorator to register a gRPC command.

        Parameters
        ----------
        name :
            Optional gRPC command name. When left undefined,
            it should be detected from the wrapped function mame.
        request_name :
            Name of the gRPC request message to be used as input.
            Defaults to the command name with a ``Request`` suffix.
        response_name :
            Name of the gRPC response message to be used as output.
            Defaults to the command name with a ``Response`` suffix.
        output :
            Expected output type of the wrapped callable.
            With ``auto``, it's inferred from output type directly;
            ``bypass`` should be used only while migrating legacy code
            as it won't wrap the output as a response message;
            ``single`` means the single output should be wrapped
            as the first field of the response message
            unless it's ``None``, in which case the response is empty;
            ``tuple`` means the result is an iterable (ideally a tuple)
            with properly ordered values that match the field order
            of the response message starting from its first field;
            ``dict`` means the output is a JSON-like Python dictionary
            (i.e., a dictionary whose keys are always strings).
            This is based on ``google.protobuf.json_format.ParseDict``,
            check its documentation for nesting rules and more details.
        cast :
            Mapping of argument names (usually input request fields)
            to convert (type cast) by means of callback routines,
            single positional input callables called once per gRPC call
            and whose results will replace their respective inputs.
        """
        cast = cast or {}

        def decorator(func):
            """Register a gRPC command."""
            snake_name = func.__name__
            command_name = name or snake2pascal(snake_name)
            if command_name in self.commands:
                raise AlreadyRegisteredHandler(command_name)
            if snake_name in self.name_map:
                raise AlreadyRegisteredName(snake_name)

            # Wrapped callable VS request message signature handling
            input_msg_name = request_name or command_name + "Request"
            request_cls = getattr(self._protos, input_msg_name)
            input_names = {rdf.name for rdf in request_cls.DESCRIPTOR.fields}
            sig = signature(func)
            sig_params = set(sig.parameters)
            missing_names = input_names - sig_params
            if missing_names:
                raise SignatureMismatch(f"Missing {missing_names} arguments")
            extra_names = sig_params - input_names
            keep_request = "request" in extra_names
            keep_context = "context" in extra_names

            # Response wrapping handling
            # (Using type instead of isinstance mostly for performance)
            output_msg_name = response_name or command_name + "Response"
            response_cls = getattr(self._protos, output_msg_name)
            output_names = [rdf.name for rdf in response_cls.DESCRIPTOR.fields]
            dict_to_resp = partial(self.typecast.py2proto, response_cls)
            type_map = {
                dict: dict_to_resp,
                tuple: lambda data: dict_to_resp(
                    dict(zip(output_names, data)),
                ),
                response_cls: lambda data: data,  # Don't wrap
            }

            def single_wrap(data):
                dict_result = {} if data is None else {output_names[0]: data}
                return dict_to_resp(dict_result)

            if output == "single":
                response_wrapper = single_wrap
            elif output == "dict":
                response_wrapper = type_map[dict]
            elif output == "tuple":
                response_wrapper = type_map[tuple]
            elif output == "bypass":
                response_wrapper = type_map[response_cls]
            else:
                def response_wrapper(data):
                    return type_map.get(type(data), single_wrap)(data)

            @wraps(func)  # Can't keep signature, but can keep __wrapped__ ref
            def wrapper(request, context):
                kwargs = self.typecast.proto2py(request)
                if keep_request:
                    kwargs["request"] = request
                if keep_context:
                    kwargs["context"] = context
                try:
                    for key, processor in cast.items():
                        kwargs[key] = processor(kwargs[key])
                    return response_wrapper(func(**kwargs))
                except Exception as exc:
                    for exc_type, handler in self.exception_handlers.items():
                        if isinstance(exc, exc_type):
                            return response_wrapper(handler(exc))
                    raise

            setattr(self._implementation, command_name, staticmethod(wrapper))
            wrapper.__name__ = command_name
            wrapper.request_cls = request_cls
            wrapper.response_cls = response_cls
            self.commands[command_name] = wrapper
            self.name_map[snake_name] = command_name
            return wrapper

        return decorator

    # TODO: check if this can be better implemented as an interceptor
    def exception_handler(
        self,
        *exceptions: type[Exception],
    ):
        """
        Parametrized decorator to register a gRPC exception handler.

        This will replace the exceptions from handling the gRPC command
        by custom messages returned by the handler.

        Positional arguments are the exceptions to be handled
        by the decorated function, whose first and only input
        will be the exception object.
        """
        def decorator(func):
            for exc_type in exceptions:
                if exc_type in self.exception_handlers:
                    raise AlreadyRegisteredHandler(exc_type.__name__)
            handlers = {
                **self.exception_handlers,
                **{exc_type: func for exc_type in exceptions},
            }

            # Sort the handlers based on their MRO in order to check
            # the most specific class handler first; not efficient,
            # but this happens only once per handler in import time,
            # and there shouldn't be many registered at once
            by_name = attrgetter("__name__")
            ordered_handlers = {}
            missed = set(handlers)
            while missed:
                mros_gen = (exc_type.mro()[1:] for exc_type in missed)
                independent_types = missed - set().union(*mros_gen)
                for exc_type in sorted(independent_types, key=by_name):
                    ordered_handlers[exc_type] = handlers[exc_type]
                missed -= independent_types

            self.exception_handlers = ordered_handlers
            return func
        return decorator

    def run(
        self,
        *,
        port: int = 443,
        max_workers: int | None = None,
        host: str = "[::]",
    ) -> None:
        # TODO: Add customization capabilities
        server = grpc.server(ThreadPoolExecutor(max_workers=max_workers))
        self.binder(self._implementation, server)
        server.add_insecure_port(f"{host}:{port}")
        server.start()
        server.wait_for_termination()


class GrpcTestClient:
    """
    Test client for keyword-only arguments for request fields,
    returning complete ``AttrDict`` instance with response fields.

    - Item access get the RPC command handler from the protobuf name,
      like ``response = client["RpcCommand"](**request_kwargs)``
    - Attribute access attempts to find the RPC command from a
      ``snake_case`` name, which should in most cases match the handler
      function name in the server implementation; it can be used like
      ``response = client.rpc_command(**request_kwargs)``
    - Defaults from request fields are auto-filled, the input arguments
      don't need to be complete
    """

    def __init__(
        self,
        server: GrpcServer,
        *,
        context=None,
        keep_server_names=False,
    ):
        self.server = server
        self.context = context  # TODO: Add a sensible default for this
        if keep_server_names:
            for snake_name, command_name in server.name_map.items():
                setattr(self, snake_name, self[command_name])

    def __getattr__(self, name):
        return self[snake2pascal(name)]

    def __getitem__(self, name):
        func = self.server.commands[name]

        @wraps(func)
        def wrapper(**kwargs):
            caster = self.server.typecast
            request = caster.py2proto(func.request_cls, kwargs)
            response = func(request, self.context)
            assert isinstance(response, func.response_cls)
            return caster.proto2py(response)

        return wrapper
