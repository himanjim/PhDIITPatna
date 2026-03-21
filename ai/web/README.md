# Web-side Utilities

This directory contains browser-side and browser-adjacent utilities used to study image compression effects and related preprocessing behaviour.

## Contents
- `browser_image_compressor.html` is the local browser tool for resizing and compressing images.
- `browser_compressed_images_accuracy_comparator.py` evaluates browser-compressed images against originals using DeepFace.
- `browser_compressed_images_insight_accuracy_comparator.py` performs a similar comparison using Triton-served InsightFace embeddings.

## Scope
Keep browser-side experimental utilities here. General-purpose backend services do not belong in this directory.