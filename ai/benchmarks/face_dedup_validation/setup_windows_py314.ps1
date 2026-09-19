$ErrorActionPreference = "Stop"

Write-Host "=== InsightFace 2.0 / raccoon_l research environment ==="
python --version

# Keep the research environment isolated in a virtual environment.  Updating
# packaging tools first avoids old pip/setuptools behaviour on a new Python.
python -m pip install --upgrade pip setuptools wheel

# InsightFace installs the CPU ONNX Runtime package as a dependency.  Install
# the rest of the stack first, then replace that CPU runtime with the NVIDIA
# build exactly as the InsightFace 2.0 documentation recommends.
python -m pip install insightface==2.0 faiss-cpu==1.15.0 numpy==2.5.3 opencv-python-headless==5.0.0.93 pandas tqdm gdown

python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install "onnxruntime-gpu[cuda,cudnn]==1.29.0"

Write-Host ""
Write-Host "Installed core versions:"
python -c "import sys, importlib.metadata as m; print('Python:',sys.version.split()[0]); print('InsightFace:',m.version('insightface')); print('ONNX Runtime GPU:',m.version('onnxruntime-gpu')); print('FAISS CPU:',m.version('faiss-cpu')); print('NumPy:',m.version('numpy'))"
