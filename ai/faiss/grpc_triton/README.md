# FAISS gRPC and Triton Stack

This directory contains the gRPC-based FAISS search service, the optional microbatching aggregator, protocol definitions, stub-generation scripts, tests, and benchmark drivers used for Triton-plus-FAISS experiments.

## Contents
- `dedup.proto` defines the core FAISS search and ingest contract.
- `dedup_faiss_grpc_bench.proto` defines the benchmark-oriented variant of the protocol.
- `gen_proto.sh` and `gen_proto_faiss_grpc_bench.sh` generate Python protobuf and gRPC stubs.
- `faiss_grpc_server.py` is the main gRPC FAISS service.
- `faiss_aggregator_grpc.py` is the microbatching aggregator in front of the service.
- `faiss_grpc_bench.py` benchmarks the gRPC path.
- `bench_faiss_service_100k.py` runs the 100k-style benchmark against the gRPC service.
- `triton_batch_bench.py` benchmarks Triton batch behaviour.
- `test_faiss_unit.py` and `test_faiss_clip_embedding.py` cover correctness and integration checks.

## Scope
This directory is the correct home for the FAISS gRPC/Triton subsystem. It contains more than benchmarks, which is why it should remain under `faiss/` rather than `benchmarks/`.