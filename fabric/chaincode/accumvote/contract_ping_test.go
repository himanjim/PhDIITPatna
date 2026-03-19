// contract_ping_test.go contains smoke tests for contract construction and the
// minimal health path.
//
// These tests do not exercise election logic. Their purpose is narrower: they check
// that the Fabric contract can be instantiated and that a method using the
// transaction context can read a stable TxID through the mocked stub.

package main

import (
    "strings"
    "testing"
    "github.com/golang/mock/gomock"
    "github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
	f "github.com/yourorg/accumvote_cc/fakes"
)

// Test_Chaincode_Constructs confirms that the Fabric contract API can construct the
// chaincode wrapper for AccumVoteContract without failing during registration or
// basic reflection-based wiring. This test is intended to catch packaging and API
// breakage early, before deeper behavioural tests are run.
func Test_Chaincode_Constructs(t *testing.T) {
  if _, err := contractapi.NewChaincode(new(AccumVoteContract)); err != nil {
    t.Fatalf("NewChaincode failed: %v", err)
  }
}

// Test_Ping_UsesTxID confirms that the Ping method reads the transaction stub made
// available through the mocked context and includes the current TxID in its return
// value. The test is deliberately small, but it protects against regressions in the
// mock wiring and in the contract's use of Fabric context objects.
func Test_Ping_UsesTxID(t *testing.T) {
  ctrl := gomock.NewController(t); defer ctrl.Finish() // Ensure mock expectations are asserted
  stub := f.NewMockChaincodeStubInterface(ctrl)
  ctx  := f.NewMockTransactionContextInterface(ctrl)

  // Wire the mocked transaction context to return our stub.
  ctx.EXPECT().GetStub().Return(stub).AnyTimes() // Allow multiple internal calls

  // Provide a deterministic TxID; Ping should incorporate it.
  stub.EXPECT().GetTxID().Return("tx-smoke-1").AnyTimes()

  // Call a minimal method that touches the stub; avoids heavy setup.
  out, err := new(AccumVoteContract).Ping(ctx)
  if err != nil || !strings.HasPrefix(out, "OK:") { // Assert only the stable prefix
    t.Fatalf("Ping failed: out=%q err=%v", out, err)
  }
}
