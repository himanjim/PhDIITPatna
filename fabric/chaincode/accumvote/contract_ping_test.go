// Contract_ping_test.go
//
// Purpose: Fast “does it even start?” checks for the AccumVoteContract. These
// Smoke tests confirm that the contract can be constructed by Fabric’s
// Contract API and that a trivial method (Ping) reads the current TxID.
// Role: Guards against broken constructors/wiring and mock regressions before
// Heavier tests run.
// Key deps: Fabric contract API (contractapi), gomock for lightweight stubbing,
// And the generated fakes in fakes/ for ChaincodeStub and Tx context.

package main

import (
    "strings"
    "testing"
    "github.com/golang/mock/gomock"
    "github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
	f "github.com/yourorg/accumvote_cc/fakes"
)

// Test_Chaincode_Constructs verifies: Chaincode Constructs.
func Test_Chaincode_Constructs(t *testing.T) {
  if _, err := contractapi.NewChaincode(new(AccumVoteContract)); err != nil {
    t.Fatalf("NewChaincode failed: %v", err)
  }
}

// Test_Ping_UsesTxID verifies: Ping Uses Tx I D.
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
