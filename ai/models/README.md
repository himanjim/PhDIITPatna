# Models

This directory contains small model-conversion or model-preparation helpers used by the AI component.

## Contents
- `convert_w600k_r50_to_dynamic_onnx.py` rewrites the ONNX graph metadata so that the batch dimension becomes dynamic.

## Scope
Keep this directory limited to model-preparation helpers. Runtime inference code should stay with the consuming subsystem.
