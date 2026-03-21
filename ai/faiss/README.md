# FAISS De-duplication Module

This directory contains the FAISS-based de-duplication services, tests, benchmark drivers, and related submodules.

## Main files
- `faiss_service.py` is the main internal HTTP FAISS service and should be treated as the canonical implementation in this directory.
- `faiss_ms.py` is a benchmark-oriented FAISS service retained because existing setup notes still refer to it.
- `bench_faiss_service_100k.py` benchmarks the HTTP FAISS service.
- `faiss_index_test.py` contains FAISS index testing logic.
- `test_faiss_unit.py` contains API and service-level tests.
- `test_faiss_clip_embedding.py` contains the optional real-embedding integration test.

## Subdirectories
- `grpc_triton/` contains the gRPC/Triton FAISS stack, including protocol definitions, servers, tests, and benchmark tools.
- `legacy/` contains older experimental FAISS services retained only for reference.

## Scope
This directory should contain all FAISS search and de-duplication code. When adding new FAISS-related work, prefer a subdirectory here rather than placing it under `benchmarks/`.