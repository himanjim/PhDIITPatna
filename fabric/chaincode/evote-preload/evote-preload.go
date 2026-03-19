/*
evote-preload.go provides the preload and lookup contract used to seed election
reference data before voting begins.

The contract has two distinct responsibilities. First, it maintains the public
candidate catalogue for each constituency, including a sorted constituency-level
candidate list used by downstream tally logic. Second, it stores presence-only
voter-roll entries in the voter_roll_pdc private data collection, allowing other
contracts to verify whether a pseudonymous voter identifier exists without
revealing the underlying private value. The design is intentionally minimal and
supports both bulk administrative preload and read-only cross-chaincode queries.
*/

package main

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
// Candidate is the public reference record for one contesting candidate in one
// constituency. The structure is stored in world state and is intended to remain
// free of any voter-linked information.
type Candidate struct {
	CandidateID   string `json:"candidate_id"`
	CandidateName string `json:"candidate_name"`
	PartyCode     string `json:"party_code"`
	SymbolHash    string `json:"symbol_hash"`
}

// VoterPresence stores only presence and status information for a pseudonymous
// voter identifier. It is deliberately minimal so that eligibility lookups can be
// performed without exposing richer electoral-roll attributes on-chain.
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

// UpsertCandidates replaces the effective candidate catalogue for one constituency.
//
// The method writes or updates each supplied candidate record, rebuilds the sorted
// constituency candidate list, and removes stale candidate entries that were
// present earlier but are absent from the new payload. It is therefore suitable
// for repeated administrative preload runs and for controlled correction of the
// candidate list before polling.
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

// PutVoterRollChunk writes a batch of presence-only voter records into
// voter_roll_pdc.
//
// The preferred input path is transient["entries"], which is expected to contain a
// raw JSON array of voter objects. The method normalises the state and
// constituency scope from the function arguments, assigns a default status of
// "eligible" when no status is supplied, and stores each pseudonymous voter key as
// a separate private-data entry.
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

// GetCandidateList returns the stored constituency candidate list as JSON.
//
// The list is expected to be the sorted list produced by UpsertCandidates. When no
// candidate list has been written for the constituency, the method returns an
// empty JSON array rather than failing.
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

// HasVoter checks whether a pseudonymous voter identifier is present in
// voter_roll_pdc for the given constituency.
//
// The method uses GetPrivateDataHash so that caller contracts can confirm roll
// membership without reading the private voter-presence record itself.
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

// main registers and starts the preload chaincode.
func main() {
	cc, err := contractapi.NewChaincode(new(Contract))
	if err != nil {
		panic(err)
	}
	if err := cc.Start(); err != nil {
		panic(err)
	}
}

