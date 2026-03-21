# Booth metadata chaincode

This directory contains the compact private-data contract for booth and device metadata.

## Contents
- `boothpdc.go`: booth preload, existence check, and read-path contract.

## Notes
- Keep this contract narrow and aligned with the validation fields consumed by `accumvote`.