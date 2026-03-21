# PhDIITPatna

This repository contains the implementation material, benchmark code, configuration files, experimental artefacts, and supporting documentation for the PhD work at IIT Patna on AI- and blockchain-enabled internet voting.

The repository brings together three connected strands of the prototype. The first is the AI component, which covers face verification, liveness assessment, FAISS-based de-duplication, and related benchmarking. The second is the Hyperledger Fabric component, which contains the chaincode, network definitions, setup material, and benchmark harnesses used to study the blockchain layer. The third is the frontend component, which provides a demonstration interface for selected user flows and API-performance measurements.

## Repository layout

- `ai/` contains the AI-side codebase, including face-verification utilities, liveness services, FAISS search and de-duplication modules, calibration scripts, benchmark drivers, and technical runbooks.
- `fabric/` contains the blockchain-side implementation, including chaincode, network configurations for Raft and SmartBFT deployments, synthetic-data utilities, benchmark scripts, setup notes, and captured benchmark results.
- `frontend/` contains the UI demonstration application and its associated documentation, tools, and performance artefacts.

## Nature of the repository

This is a research and implementation repository rather than a polished product distribution. It therefore combines executable code with experiment notes, command runbooks, intermediate utilities, and selected result files. Some directories are structured as self-contained subprojects, while others reflect the practical workflow of an active PhD prototype.

The repository should be read with that purpose in mind. It is intended to preserve the technical development of the proposed system, support benchmarking and comparative evaluation, and document the choices made during implementation.

## How to use this repository

Each major subtree should be approached through its own `README.md` file. Those local README files describe the scope, expectations, and conventions of the relevant component more accurately than a single top-level note can do.

In general:

- treat `docs/` directories as the primary location for setup notes and runbooks;
- treat `results/`, `perf/`, and similar folders as stored artefacts rather than active source code;
- keep new code within the existing domain structure unless there is a clear technical reason to introduce a new directory;
- make conservative structural changes, especially where filenames are already cited in scripts, runbooks, or benchmark notes.

## Scope and limitations

The repository is suitable as a technical research workspace and implementation record. It is not yet a uniform production-style monorepo. Some parts still depend on local environment assumptions, document-based setup steps, or script-oriented execution patterns. That is a normal consequence of the repository’s role in an evolving research programme, but it should be recognised by anyone attempting to reproduce or extend the work.

## Purpose

The purpose of this repository is to support the design, implementation, and evaluation of a secure and scalable internet voting framework for the Indian context. It captures the software artefacts, technical experiments, and supporting material needed to study the integration of AI-based voter-side verification with a blockchain-backed voting backend and a demonstrative user interface.