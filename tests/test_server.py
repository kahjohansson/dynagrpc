import pytest

from dynagrpc import GrpcServer, GrpcTestClient


@pytest.mark.parametrize("first,second,expected",
                         [(5, 7, 12), (6, -9, -3), (-1, -3, -4)])
def test_add(first, second, expected):
    server = GrpcServer("tests", "Maths", "maths")

    @server.rpc()
    def add(first, second):
        return first + second

    client = GrpcTestClient(server)
    assert client.add(first=first, second=second) == {"result": expected}
