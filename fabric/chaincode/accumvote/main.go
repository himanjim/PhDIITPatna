package main

import (
    "log"
    "github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

func main() {
    cc, err := contractapi.NewChaincode(new(AccumVoteContract))
    if err != nil {
        log.Panicf("error creating accumvote_cc: %v", err)
    }
    if err := cc.Start(); err != nil {
        log.Panicf("error starting accumvote_cc: %v", err)
    }
}
