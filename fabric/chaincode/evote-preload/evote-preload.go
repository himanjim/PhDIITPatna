package main

/*
evote-preload (bootstrap/minimal)

Exports (for admin/preload & verification only):
  1) UpsertCandidates(constituencyID, candidatesJSON)
       PUBLIC state:
         CAND::<constituencyID>::<candidateID>     → full Candidate JSON
         CANDIDX::<constituencyID>::<candidateID>  → "1"
         CANDLIST::<constituencyID>                → ["cand-000001", ...] (sorted)
     - Idempotent and *prunes* stale entries that are no longer present.

  2) PutVoterRollChunk(stateCode, constituencyID, entriesJSON_or_empty)
       PRIVATE DATA COLLECTION "voter_roll_pdc"
       • Preferred: pass entries via transient map key "entries" (JSON array bytes)
       • Fallback: if transient missing, uses the third arg (for quick tests)
       entries := [{"voter_id_star":"...", "status":"eligible"}, ...]

  3) GetCandidateList(constituencyID) → JSON list of candidate IDs
  4) HasVoter(constituencyID, voter_id_star) → "true"/"false" (via GetPrivateDataHash)
*/

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// -----------------------------------------------------------------------------
// Keys
// -----------------------------------------------------------------------------

func candKey(constituencyID, candidateID string) string    { return "CAND::" + constituencyID + "::" + candidateID }
func candIdxKey(constituencyID, candidateID string) string { return "CANDIDX::" + constituencyID + "::" + candidateID }
func candListKey(constituencyID string) string             { return "CANDLIST::" + constituencyID }
func voterKey(constituencyID, voterStar string) string     { return "VOTER::" + constituencyID + "::" + voterStar }

// -----------------------------------------------------------------------------
// Models
// -----------------------------------------------------------------------------

type Candidate struct {
	CandidateID   string `json:"candidate_id"`
	CandidateName string `json:"candidate_name"`
	PartyCode     string `json:"party_code"`
	SymbolHash    string `json:"symbol_hash"`
}

type VoterPresence struct {
	Status       string `json:"status"`
	StateCode    string `json:"state_code"`
	Constituency string `json:"constituency_id"`
}

// -----------------------------------------------------------------------------
// Contract
// -----------------------------------------------------------------------------

type Contract struct {
	contractapi.Contract
}

// UpsertCandidates loads/updates the candidate catalogue for a constituency.
// It is idempotent and also *removes* candidates that are no longer in the list.
func (c *Contract) UpsertCandidates(ctx contractapi.TransactionContextInterface, constituencyID string, candidatesJSON string) error {
	constituencyID = strings.TrimSpace(constituencyID)
	if constituencyID == "" {
		return fmt.Errorf("constituencyID empty")
	}

	// Parse payload
	var rows []Candidate
	if err := json.Unmarshal([]byte(candidatesJSON), &rows); err != nil {
		return fmt.Errorf("parse candidates: %w", err)
	}

	// Build new set
	newIDs := make([]string, 0, len(rows))
	newSet := make(map[string]struct{}, len(rows))

	for _, r := range rows {
		r.CandidateID = strings.TrimSpace(r.CandidateID)
		r.CandidateName = strings.TrimSpace(r.CandidateName)
		r.PartyCode = strings.TrimSpace(r.PartyCode)
		r.SymbolHash = strings.TrimSpace(r.SymbolHash)

		if r.CandidateID == "" {
			return fmt.Errorf("candidate entry missing candidate_id")
		}
		newIDs = append(newIDs, r.CandidateID)
		newSet[r.CandidateID] = struct{}{}

		b, _ := json.Marshal(r)
		if err := ctx.GetStub().PutState(candKey(constituencyID, r.CandidateID), b); err != nil {
			return err
		}
		if err := ctx.GetStub().PutState(candIdxKey(constituencyID, r.CandidateID), []byte("1")); err != nil {
			return err
		}
	}

	// Prune stale candidates (present earlier but missing now)
	oldListBytes, err := ctx.GetStub().GetState(candListKey(constituencyID))
	if err != nil {
		return err
	}
	if len(oldListBytes) > 0 {
		var oldIDs []string
		_ = json.Unmarshal(oldListBytes, &oldIDs)
		for _, old := range oldIDs {
			if _, still := newSet[old]; !still {
				if err := ctx.GetStub().DelState(candKey(constituencyID, old)); err != nil {
					return err
				}
				if err := ctx.GetStub().DelState(candIdxKey(constituencyID, old)); err != nil {
					return err
				}
			}
		}
	}

	// Write sorted list (even if empty)
	sort.Strings(newIDs)
	bList, _ := json.Marshal(newIDs)
	return ctx.GetStub().PutState(candListKey(constituencyID), bList)
}

// PutVoterRollChunk stores presence-only voter records into the PDC.
// Preferred input is transient map key "entries" (raw JSON array).
func (c *Contract) PutVoterRollChunk(ctx contractapi.TransactionContextInterface, stateCode, constituencyID, entriesJSON string) error {
	stateCode = strings.TrimSpace(stateCode)
	constituencyID = strings.TrimSpace(constituencyID)
	if stateCode == "" || constituencyID == "" {
		return fmt.Errorf("stateCode/constituencyID empty")
	}

	// Try transient first
	if tmap, err := ctx.GetStub().GetTransient(); err == nil {
		if b, ok := tmap["entries"]; ok && len(b) > 0 {
			entriesJSON = string(b) // bytes are already the JSON array
		}
	}

	// Parse entries
	var entries []map[string]string
	if err := json.Unmarshal([]byte(entriesJSON), &entries); err != nil {
		return fmt.Errorf("parse entries: %w", err)
	}
	if len(entries) == 0 {
		return nil
	}

	for _, e := range entries {
		voterStar := strings.TrimSpace(e["voter_id_star"])
		status := strings.TrimSpace(e["status"])
		if voterStar == "" {
			return fmt.Errorf("voter_id_star missing in an entry")
		}
		// tiny validation; expand if you add more statuses
		if status == "" {
			status = "eligible"
		}

		vp := VoterPresence{
			Status:       status,
			StateCode:    stateCode,
			Constituency: constituencyID,
		}
		b, _ := json.Marshal(vp)

		if err := ctx.GetStub().PutPrivateData("voter_roll_pdc", voterKey(constituencyID, voterStar), b); err != nil {
			return err
		}
	}
	return nil
}

// GetCandidateList returns the sorted candidate IDs for a constituency.
func (c *Contract) GetCandidateList(ctx contractapi.TransactionContextInterface, constituencyID string) (string, error) {
	constituencyID = strings.TrimSpace(constituencyID)
	if constituencyID == "" {
		return "", fmt.Errorf("constituencyID empty")
	}
	b, err := ctx.GetStub().GetState(candListKey(constituencyID))
	if err != nil {
		return "", err
	}
	if len(b) == 0 {
		return "[]", nil
	}
	return string(b), nil
}

// HasVoter returns true if a (hashed/pseudonymous) voter key exists in the PDC.
// Uses GetPrivateDataHash (works without exposing the private value).
func (c *Contract) HasVoter(ctx contractapi.TransactionContextInterface, constituencyID, voterStar string) (bool, error) {
	constituencyID = strings.TrimSpace(constituencyID)
	voterStar = strings.TrimSpace(voterStar)
	if constituencyID == "" || voterStar == "" {
		return false, fmt.Errorf("constituencyID/voter_id_star empty")
	}
	h, err := ctx.GetStub().GetPrivateDataHash("voter_roll_pdc", voterKey(constituencyID, voterStar))
	if err != nil {
		return false, err
	}
	return len(h) > 0, nil
}

func main() {
	cc, err := contractapi.NewChaincode(new(Contract))
	if err != nil {
		panic(err)
	}
	if err := cc.Start(); err != nil {
		panic(err)
	}
}

