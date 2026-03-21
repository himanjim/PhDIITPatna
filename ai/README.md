# AI Component

This directory contains the AI-side code and supporting documents used in the PhD prototype for face verification, liveness detection, FAISS-based de-duplication, benchmarking, and browser-side validation utilities.

## Structure
- `benchmarks/` contains standalone benchmarking and validation scripts.
- `calibration/` contains threshold-calibration utilities.
- `docs/` contains setup notes, command runbooks, and experiment support documents.
- `face/` contains the liveness-verification pipeline, related services, and clip-based test tools.
- `faiss/` contains FAISS-based de-duplication services, tests, and the gRPC/Triton FAISS stack.
- `models/` contains model-conversion helpers.
- `servers/` contains integration services that connect embedding generation to search.
- `web/` contains browser-side compression and image-comparison utilities.

## Notes
- `faiss/faiss_service.py` should be treated as the main internal HTTP FAISS service.
- `faiss/faiss_ms.py` remains a benchmark-oriented service variant because existing runbooks still refer to it.
- `faiss/legacy/` is reserved for older experimental services kept only for reference.
- `faiss/grpc_triton/` contains the gRPC/Triton FAISS service stack and related benchmarks.

## Practical guidance
This component is script-oriented rather than package-oriented. Before adding new code, prefer to place it under the existing domain directory instead of creating a new top-level folder. New operational notes should go under `docs/`, not beside code.