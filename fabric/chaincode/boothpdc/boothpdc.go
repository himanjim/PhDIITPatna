// boothpdc.go implements a compact registry for polling-booth and polling-device
// metadata stored in the booth_pdc private data collection.
//
// The contract is intentionally narrow. It supports bulk preload of booth records,
// existence checks via private-data hashes, and full reads for authorised peers.
// The stored record binds booth identity, operating window, officer assignment,
// and device fingerprinting fields that are later used by the voting contract for
// booth- and device-level validation during tally or audit.

package main

import (
  "encoding/json"
  "errors"
  "fmt"
  "strings"
  "github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

const boothColl = "booth_pdc"

// BoothRecord is the canonical private record for one polling booth within one
// constituency. It combines operational status, polling window, officer metadata,
// and device linkage fields so that downstream contracts can validate whether a
// ballot was cast from an authorised booth-device context.
type BoothRecord struct {
  StateCode            string `json:"state_code"`
  ConstituencyID       string `json:"constituency_id"`
  BoothID              string `json:"booth_id"`
  Status               string `json:"status"` // "A" or "X"
  OpenTime             int64  `json:"open_time"`
  CloseTime            int64  `json:"close_time"`
  OfficerID            string `json:"officer_id"`
  OfficerRole          string `json:"officer_role"`
  DeviceID             string `json:"device_id"`
  DeviceKeyFingerprint string `json:"device_key_fingerprint"`
}

type SmartContract struct{ contractapi.Contract }

// keyBooth constructs the deterministic private-data key for one booth record.
// The key is namespaced by state, constituency, and booth so that the same booth
// identifier can be reused safely across different jurisdictions.
func keyBooth(sc, cid, bid string) string {
  return "BOOTH::" + sc + "::" + cid + "::" + bid
}

// PutBoothChunk bulk-loads booth records into booth_pdc from transient input.
//
// The caller supplies stateCode and constituencyID as method arguments and passes
// the record array in transient["entries"] as raw JSON bytes. For each record, the
// method fills missing state or constituency values from the method arguments,
// normalises booth status, rejects cross-constituency mismatches, and then writes
// the resulting record to private data under a deterministic composite key.
func (s *SmartContract) PutBoothChunk(ctx contractapi.TransactionContextInterface, stateCode, constituencyID string) error {
  if stateCode == "" || constituencyID == "" { return errors.New("state_code and constituency_id are required") }
  tm, err := ctx.GetStub().GetTransient(); if err != nil { return fmt.Errorf("get transient: %w", err) }
  raw, ok := tm["entries"]; if !ok || len(raw) == 0 { return errors.New("transient[entries] missing") }
  var recs []BoothRecord
  if err := json.Unmarshal(raw, &recs); err != nil { return fmt.Errorf("decode entries: %w", err) }
  for _, r := range recs {
    if r.StateCode == "" { r.StateCode = stateCode }
    if r.ConstituencyID == "" { r.ConstituencyID = constituencyID }
    if r.StateCode != stateCode || r.ConstituencyID != constituencyID {
      return fmt.Errorf("row state/cid mismatch for booth %q", r.BoothID)
    }
    if r.BoothID == "" { return errors.New("booth_id empty") }
    r.Status = strings.ToUpper(strings.TrimSpace(r.Status))
    if r.Status != "A" && r.Status != "X" {
      return fmt.Errorf("invalid status %q for %s/%s/%s", r.Status, r.StateCode, r.ConstituencyID, r.BoothID)
    }
    val, err := json.Marshal(r); if err != nil { return fmt.Errorf("marshal: %w", err) }
    if err := ctx.GetStub().PutPrivateData(boothColl, keyBooth(r.StateCode, r.ConstituencyID, r.BoothID), val); err != nil {
      return fmt.Errorf("put PDC: %w", err)
    }
  }
  return nil
}

// HasBooth performs an existence check without reading the private booth value.
//
// It relies on GetPrivateDataHash so that a peer can confirm whether a booth entry
// is present in booth_pdc while avoiding disclosure of the stored booth record.
func (s *SmartContract) HasBooth(ctx contractapi.TransactionContextInterface, stateCode, constituencyID, boothID string) (bool, error) {
  h, err := ctx.GetStub().GetPrivateDataHash(boothColl, keyBooth(stateCode, constituencyID, boothID))
  if err != nil { return false, err }
  return len(h) > 0, nil
}

// GetBooth returns the full private booth record for an authorised caller.
//
// This method is intended for administrative validation and audit workflows that
// need the stored booth metadata itself, rather than only proof that a record
// exists.
func (s *SmartContract) GetBooth(ctx contractapi.TransactionContextInterface, stateCode, constituencyID, boothID string) (string, error) {
  val, err := ctx.GetStub().GetPrivateData(boothColl, keyBooth(stateCode, constituencyID, boothID))
  if err != nil { return "", err }
  if len(val) == 0 { return "", fmt.Errorf("booth not found") }
  return string(val), nil
}

// main registers and starts the booth metadata chaincode.
func main() {
  cc, err := contractapi.NewChaincode(new(SmartContract)); if err != nil { panic(err) }
  if err := cc.Start(); err != nil { panic(err) }
}
