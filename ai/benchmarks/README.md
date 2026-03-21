# Benchmarks

This directory contains standalone benchmark and validation scripts that are not part of the main service implementations.

## Contents
- `deepface_benchmark_single_script.py` benchmarks a DeepFace-plus-FAISS workflow in one script.
- `deep_server_benchmark.py` performs a small-scale embedding benchmark against local images.
- `deepface_verification_config_benchmark.py` compares DeepFace model, detector, and metric combinations.
- `deepface_verification_config_benchmark_parallel.py` runs a parallelised version of the same comparison workflow.
- `deepface_faiss_http_benchmark_client.py` and `deepface_faiss_http_benchmark_server.py` form a simple HTTP benchmark pair.
- `locust_benchmark.py` is the Locust-based load generator.
- `system_validation_test.py` is a validation-oriented benchmark or end-to-end check.
- `extract_docx_bracket_references.py` is a small utility script retained for document-related extraction work.

## Scope
This directory should contain benchmark drivers, validation scripts, and one-off measurement utilities only. Service implementations, protocol definitions, and long-lived infrastructure code should live in their own functional directories.
