# Frontend services

This directory contains service-layer code used by the UI demo.

## Contents
- `api.ts`: API-facing calls and request wrappers.
- `mockBackend.ts`: in-browser mock backend used by the demo.

## Notes
- This directory is the right place for service abstractions; keep transport logic out of route components where possible.