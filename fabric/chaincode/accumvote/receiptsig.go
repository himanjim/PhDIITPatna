// receiptsig.go defines the signed receipt.
//
// A voter whose receipt later returns unknown_tx can show that something went
// wrong, but without a signature cannot prove that the system issued that
// receipt. The gateway therefore signs, with an Ed25519 key whose public half
// (RECEIPT_SIGNING_KEY) is published before polling, the fields that identify
// the recorded ballot. The signature proves that a ballot with commitment hC
// was accepted; it says nothing about the choice, so receipt-freeness is
// unchanged.
//
// The contract does not call this file: the gateway signs, and Client B and the
// public receipt check verify. Like packed.go it has no Fabric dependency.
package main

import (
	"crypto/ed25519"
	"encoding/hex"
	"fmt"
	"strings"
)

const receiptDomain = "RECEIPT_V1"

// ReceiptFields are the signed fields, as printed on the receipt.
type ReceiptFields struct {
	EID      string
	Serial   string // or the receipt reference rid in the stronger privacy profile
	TxID     string
	HC       string
	Epoch    string
	CastTime string
}

// ReceiptMessage is the byte string the gateway signs.
func ReceiptMessage(f ReceiptFields) []byte {
	return []byte(strings.Join([]string{
		receiptDomain, f.EID, f.Serial, f.TxID, strings.ToLower(strings.TrimSpace(f.HC)), f.Epoch, f.CastTime,
	}, "|"))
}

// SignReceipt returns the hex signature the gateway places on the receipt.
func SignReceipt(priv ed25519.PrivateKey, f ReceiptFields) string {
	return hex.EncodeToString(ed25519.Sign(priv, ReceiptMessage(f)))
}

// VerifyReceiptSignature checks a receipt signature against the published key.
func VerifyReceiptSignature(pubKeyHex string, f ReceiptFields, sigHex string) error {
	pub, err := hex.DecodeString(strings.TrimSpace(pubKeyHex))
	if err != nil || len(pub) != ed25519.PublicKeySize {
		return fmt.Errorf("receipt signature: bad receipt signing key")
	}
	sig, err := hex.DecodeString(strings.TrimSpace(sigHex))
	if err != nil || !ed25519.Verify(pub, ReceiptMessage(f), sig) {
		return fmt.Errorf("receipt signature: does not verify")
	}
	return nil
}
