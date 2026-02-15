
"""
sitecustomize_v2.py

Rename this file to `sitecustomize.py` in your project folder.

Python auto-imports `sitecustomize` at startup (if importable). This makes your
unchanged `liveness_check.py` work even when MediaPipe has no `mp.solutions`.
"""

# Runtime hook
# ------------
# Python auto-imports `sitecustomize` at interpreter startup if it is importable.
# We use this hook to patch MediaPipe so that `mp.solutions.face_mesh.FaceMesh` exists
# for unchanged code paths.

try:
    from mp_face_mesh_tasks_shim_v2 import patch_mediapipe_solutions
    patch_mediapipe_solutions()
except Exception:
    pass
