"""
sitecustomize.py provides a startup-time compatibility hook for environments in which MediaPipe no longer exposes the legacy mp.solutions face-mesh entry points expected by the existing liveness code. Python imports sitecustomize automatically when it is importable, so this file allows the compatibility patch to be applied without modifying the original liveness_check.py call sites.
"""

# Runtime hook
# ------------
# Python auto-imports `sitecustomize` at interpreter startup if it is importable.
# We use this hook to patch MediaPipe so that `mp.solutions.face_mesh.FaceMesh` exists
# for unchanged code paths.

# The import-and-patch sequence is intentionally performed at interpreter startup so that downstream modules see a MediaPipe surface compatible with older code. This file acts as a narrow compatibility shim, not as a general-purpose application initialiser.
try:
    # Import the local shim only when the interpreter actually starts inside an environment that includes this project.            Keeping the dependency local avoids forcing unrelated Python sessions to know about the patch module.
    from mp_face_mesh_tasks_shim_v2 import patch_mediapipe_solutions

    # Apply the patch once at startup so that unchanged code can continue to resolve mp.solutions.face_mesh.FaceMesh even         when the installed MediaPipe package exposes only the newer Tasks-style interface.
    patch_mediapipe_solutions()
# Failure is swallowed deliberately. The shim is a compatibility aid rather than a mandatory runtime dependency, and suppressing startup failure preserves ordinary Python behaviour in environments where the patch module is absent or MediaPipe is not installed.
except Exception:
    pass
