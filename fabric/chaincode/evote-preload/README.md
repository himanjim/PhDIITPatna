# Preload chaincode

This directory contains the preload contract used to seed candidate lists and presence-only voter-roll entries before voting begins.

## Contents
- `evote-preload.go`: candidate upsert, voter-roll preload, and query helpers.

## Notes
- The folder name matches the deployed chaincode naming convention; change it only if you are prepared to update deployment runbooks and references.