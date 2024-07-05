from dynagrpc import GrpcServer, GrpcTestClient


def test_add():
    server = GrpcServer("tests", "Maths", "maths")

    @server.rpc()
    def add(first, second):
        return first + second

    client = GrpcTestClient(server)
    assert client.add(first=5, second=7) == {"result": 12}
