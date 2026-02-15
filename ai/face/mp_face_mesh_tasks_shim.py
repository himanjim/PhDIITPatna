
"""
mp_face_mesh_tasks_shim_v2.py

Tasks-backed shim that provides:
    mp.solutions.face_mesh.FaceMesh

This is required when the installed `mediapipe` package does not ship legacy
`mp.solutions` but your unchanged `liveness_check.py` still expects it.

Model required (download from official MediaPipe Face Landmarker page):
    face_landmarker.task

Provide it via:
  - env MP_FACE_LANDMARKER_TASK=...full path..., or
  - place at .\models\face_landmarker.task
"""

# Compatibility note
# ------------------
# Some MediaPipe distributions do not expose the legacy `mediapipe.solutions` API.
# This shim provides `mp.solutions.face_mesh.FaceMesh` by wrapping the MediaPipe Tasks
# FaceLandmarker implementation and returning a minimal result compatible with callers
# that only rely on `multi_face_landmarks`.
#
# Thread-safety: the Tasks landmarker is protected with a lock because concurrent calls
# are not guaranteed to be safe on all builds/platforms.

from __future__ import annotations

import os, sys, types, threading
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class _LM:
    x: float
    y: float
    z: float = 0.0

@dataclass
class _LandmarkList:
    landmark: List[_LM]

@dataclass
class _ProcessResult:
    multi_face_landmarks: Optional[List[_LandmarkList]]

def _resolve_task_model_path() -> str:
    env = os.environ.get("MP_FACE_LANDMARKER_TASK") or os.environ.get("MP_FACE_LANDMARKER_TASK_PATH")
    if env and os.path.isfile(env):
        return env
    local = os.path.join(os.getcwd(), "models", "face_landmarker.task")
    if os.path.isfile(local):
        return local
    raise RuntimeError(
        "Tasks FaceLandmarker model not found.\n"
        "Download `face_landmarker.task` and either set MP_FACE_LANDMARKER_TASK to its path\n"
        "or place it at .\\models\\face_landmarker.task"
    )

class FaceMesh:
    def __init__(
        self,
        static_image_mode: bool = False,  # accepted; ignored
        max_num_faces: int = 1,
        refine_landmarks: bool = False,   # accepted; ignored
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._lock = threading.Lock()
        self._ts_ms = 0
        model_path = _resolve_task_model_path()

        import mediapipe as mp  # type: ignore
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=int(max_num_faces) if max_num_faces else 1,
            min_face_detection_confidence=float(min_detection_confidence),
            min_face_presence_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mp = mp
        self._landmarker = FaceLandmarker.create_from_options(opts)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def process(self, image_rgb) -> _ProcessResult:
        mp = self._mp
        with self._lock:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            self._ts_ms += 33
            res = self._landmarker.detect_for_video(mp_image, self._ts_ms)

        faces = getattr(res, "face_landmarks", None)
        if not faces:
            return _ProcessResult(multi_face_landmarks=None)

        out: List[_LandmarkList] = []
        for face_lms in faces:
            lm_list = [_LM(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))) for lm in face_lms]
            out.append(_LandmarkList(landmark=lm_list))
        return _ProcessResult(multi_face_landmarks=out)

def patch_mediapipe_solutions() -> None:
    import mediapipe as mp  # type: ignore

    if getattr(mp, "solutions", None) is not None:
        try:
            if hasattr(mp.solutions, "face_mesh") and hasattr(mp.solutions.face_mesh, "FaceMesh"):
                return
        except Exception:
            pass

    sol_mod = types.ModuleType("mediapipe.solutions")
    face_mesh_mod = types.ModuleType("mediapipe.solutions.face_mesh")
    face_mesh_mod.FaceMesh = FaceMesh  # type: ignore[attr-defined]
    sol_mod.face_mesh = face_mesh_mod  # type: ignore[attr-defined]

    mp.solutions = sol_mod  # type: ignore[attr-defined]
    sys.modules["mediapipe.solutions"] = sol_mod
    sys.modules["mediapipe.solutions.face_mesh"] = face_mesh_mod
