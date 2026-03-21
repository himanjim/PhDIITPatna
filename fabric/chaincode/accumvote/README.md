# AccumVote chaincode

This directory contains the main voting contract together with its Go test suite.

## Contents
- `accumvote.go`: contract implementation.
- `*_test.go`: unit, integration-style, and workflow tests for voting, tallying, publication, and receipt verification.
- `harness_test.go`: shared in-memory or mocked test harness support.

## Notes
- This directory needs a Go module file and explicit dependency pinning.
- Treat the test suite as a major part of the contract documentation, not as an optional add-on.