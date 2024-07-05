from google.protobuf.json_format import (
    MessageToDict as message_to_deserialized_json,
)
import grpc

from dynagrpc import GrpcTypeCastRegistry


def test_dummy_proto_astcast():
    dummy_pb2 = grpc.protos("dummy.proto")  # Requires grpcio-tools

    # Create the converters
    caster = GrpcTypeCastRegistry()
    caster.register_enum_type(dummy_pb2.Code.DESCRIPTOR)
    caster.register_message_type(dummy_pb2.I64Range.DESCRIPTOR)
    caster.register_message_type(dummy_pb2.Dummy.DESCRIPTOR)

    # Convert dictionary to protobuf
    dict_data = {
        "ranges": [
            {"start": -1, "end": 0},
            {"start": 700_000, "end": 7_123_000},
        ],
        "codes": {
            "null": "UNSPECIFIED",
            "test": "CONCRETE",
            "tree": "ABSTRACT",
            "grpc": "BOTH",
        },
        "number": 3.14,
        "is_random": True,
    }
    proto_obj = caster.py2proto(dummy_pb2.Dummy, dict_data)

    assert isinstance(proto_obj, dummy_pb2.Dummy)
    assert proto_obj.is_random is not True  # Wrapper behavior
    assert proto_obj.is_random.value is True
    assert 3.1 < proto_obj.number < 3.2  # "optional" don't appear as wrapped
    codes_as_dict = dict(proto_obj.codes)  # Direct access use numbers for enum
    assert codes_as_dict == {"null": 0, "test": 2, "tree": 1, "grpc": 3}
    assert message_to_deserialized_json(proto_obj) == {
        "ranges": [  # JSON spec requires Int64 as strings
            {k: str(v) for k, v in range_dict.items() if v != 0}
            for range_dict in dict_data["ranges"]
        ],
        # Enums always have the prefix in JSON
        "codes": {k: "CODE_" + v for k, v in dict_data["codes"].items()},
        "number": 3.14,
        "isRandom": True,
    }
