package main

import (
  "encoding/json"
  "errors"
  "fmt"
  "strings"
  "github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

const boothColl = "booth_pdc"

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

func keyBooth(sc, cid, bid string) string {
  return "BOOTH::" + sc + "::" + cid + "::" + bid
}

// PutBoothChunk writes an array of BoothRecord into PDC via transient["entries"] = base64(JSON array)
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

func (s *SmartContract) HasBooth(ctx contractapi.TransactionContextInterface, stateCode, constituencyID, boothID string) (bool, error) {
  h, err := ctx.GetStub().GetPrivateDataHash(boothColl, keyBooth(stateCode, constituencyID, boothID))
  if err != nil { return false, err }
  return len(h) > 0, nil
}

func (s *SmartContract) GetBooth(ctx contractapi.TransactionContextInterface, stateCode, constituencyID, boothID string) (string, error) {
  val, err := ctx.GetStub().GetPrivateData(boothColl, keyBooth(stateCode, constituencyID, boothID))
  if err != nil { return "", err }
  if len(val) == 0 { return "", fmt.Errorf("booth not found") }
  return string(val), nil
}

func main() {
  cc, err := contractapi.NewChaincode(new(SmartContract)); if err != nil { panic(err) }
  if err := cc.Start(); err != nil { panic(err) }
}
