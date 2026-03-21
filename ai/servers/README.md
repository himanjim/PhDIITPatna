# Integration Servers

This directory contains lightweight service wrappers that connect embedding extraction to downstream search or verification components.

## Contents
- `deepface_server.py` exposes a minimal FastAPI service that computes a DeepFace embedding and forwards it to a downstream FAISS service.

## Scope
This directory is for small integration-oriented servers. If additional long-lived service code is added, consider grouping it by subsystem before creating more top-level folders.