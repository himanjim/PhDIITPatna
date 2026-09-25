# Benchmarks

This directory contains standalone benchmark and validation scripts that are not
part of the main service implementations, together with the face de-duplication
validation study.

## Contents

- `deepface_benchmark_single_script.py` benchmarks a DeepFace-plus-FAISS workflow in one script.
- `deep_server_benchmark.py` performs a small-scale embedding benchmark against local images.
- `deepface_verification_config_benchmark.py` compares DeepFace model, detector, and metric combinations.
- `deepface_verification_config_benchmark_parallel.py` runs a parallelised version of the same comparison workflow.
- `deepface_faiss_http_benchmark_client.py` and `deepface_faiss_http_benchmark_server.py` form a simple HTTP benchmark pair.
- `locust_benchmark.py` is the Locust-based load generator.
- `system_validation_test.py` is a validation-oriented benchmark or end-to-end check.
- `extract_docx_bracket_references.py` is a small utility script retained for document-related extraction work.

## Subdirectories

- `face_dedup_validation/` holds the face de-duplication validation study: threshold
  calibration for the `raccoon_l` pipeline, the CelebA large-gallery 1:N experiment,
  the search-level FPIR recalibration, and the PGU-Face beard and stubble robustness
  test, together with the compact result artefacts the manuscript tables are drawn
  from. It has a substantial README of its own and should be read through that file
  rather than through this one. Unlike the scripts above it is a self-contained
  study with its own `requirements.txt` and environment setup, and it uses the
  InsightFace `raccoon_l` pack and exact FAISS search rather than the operational
  Triton and TensorRT pipeline benchmarked elsewhere.

## Scope

The top level of this directory should contain benchmark drivers, validation
scripts, and one-off measurement utilities only. Service implementations,
protocol definitions, and long-lived infrastructure code should live in their own
functional directories. A multi-script study with its own environment, datasets
and result artefacts should be a subdirectory here, as
`face_dedup_validation/` is.
