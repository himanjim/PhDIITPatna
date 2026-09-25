# Face and Liveness Module

This directory contains the liveness-verification code path, clip-processing
utilities, local and gRPC service layers, and related test tools.

## Main files

- `liveness_check.py` contains the main liveness-analysis logic.
- `liveness_core.py` provides the integration layer used by services and tests.
- `lvs_service.py` exposes the HTTP service.
- `liveness_grpc_server.py` exposes the gRPC service.
- `liveness_aggregator_grpc.py` fronts multiple gRPC replicas behind a simple forwarding layer.
- `liveness.proto` defines the gRPC contract that `liveness_grpc_server.py` and
  `liveness_aggregator_grpc.py` implement. It is the clip-cache benchmarking
  interface, in which replicas preload decoded clips into memory and clients
  reference them by identifier, so that repeated media upload is removed from the
  measured path. The service exposes four calls: `Health`, `GetPolicy`,
  `ListClips` and `CheckByClipId`.
- `bench_liveness_grpc.py` benchmarks the gRPC service.
- `record_liveness_clips.py` prepares or records clip inputs.
- `test_liveness_clips.py` checks service and clip-processing behaviour.
- `mp_face_mesh_tasks_shim.py` provides MediaPipe compatibility support.
- `sitecustomize.py` applies startup-time compatibility fixes required by the liveness stack.

## Utility scripts

- `cv_face_extractor.py` extracts face crops or related image outputs.
- `image_augmentation.py` contains augmentation logic for image preparation.
- `facenet_vs_insightvit.py` compares model behaviour.
- `client_post_clip.py` exercises the HTTP endpoint from the client side.
- `liveness_check.ipynb` is the Google Colab notebook used to run
  `liveness_check.py` on a hosted GPU. It installs the dependencies and invokes
  the script, and is a convenience wrapper rather than a separate implementation.
  The authoritative logic is in `liveness_check.py`.

## Scope

Keep liveness-related logic, services, and supporting utilities together here.
New runbooks should go to `docs/`, not this directory. When the gRPC interface
changes, change `liveness.proto` first and regenerate the stubs, so that the
protocol definition remains the single description of the contract.
