# DynagRPC

DynagRPC is a Python library to help writing/using gRPC and protobuf.

# Simple server implementation

Say we have this protobuf definition for a simple addition RPC:

```protobuf
syntax = "proto3";
package tests.maths;

service Maths {
  rpc Add (AddRequest) returns (AddResponse) {}
}

message AddRequest {
  int32 first = 1;
  int32 second = 2;
}

message AddResponse {
  int32 result = 1;
}
```

Keeping just the proto (i.e., not compiled with `protoc`), we can use
it with the following implementation in Python is:

```python
from dynagrpc import GrpcServer

server = GrpcServer("tests", "Maths", "maths")

@server.rpc()
def add(first, second):
    return first + second
```

Which you can run, for example, with:

```python
server.run(host="localhost", port="50051")
```
