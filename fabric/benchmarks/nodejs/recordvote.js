'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');
const fs = require('fs');
const path = require('path');

/**
 * Load a CSV file into an array of objects using the header row as keys.
 * Simple parser: assumes no embedded commas/quotes in fields.
 */
function loadCsv(filePath) {
    const fullPath = path.resolve(filePath);
    const content = fs.readFileSync(fullPath, 'utf8');
    const lines = content.split(/\r?\n/).filter(l => l.trim().length > 0);

    if (lines.length === 0) {
        throw new Error(`CSV file is empty: ${fullPath}`);
    }

    const headers = lines[0].split(',').map(h => h.trim());
    const records = [];

    for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',');
        const obj = {};
        headers.forEach((h, idx) => {
            obj[h] = (cols[idx] || '').trim();
        });
        records.push(obj);
    }

    return records;
}

class RecordVoteWorkload extends WorkloadModuleBase {

    constructor() {
        super();
        this.txIndex = 0;
        this.voters = [];
        this.candidates = [];
        this.booths = [];
    }

    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);

        this.workerIndex = workerIndex;

        // CSV paths (can be overridden in benchconfig)
        const votersCsv     = roundArguments.votersCsv     || 'data/voters.csv';
        const candidatesCsv = roundArguments.candidatesCsv || 'data/candidates.csv';
        const boothsCsv     = roundArguments.boothsCsv     || 'data/booths.csv';

        // Load CSVs once per worker
        this.voters     = loadCsv(votersCsv);
        this.candidates = loadCsv(candidatesCsv);
        this.booths     = loadCsv(boothsCsv);

        if (this.voters.length === 0) {
            throw new Error(`No voters loaded from ${votersCsv}`);
        }
        if (this.candidates.length === 0) {
            throw new Error(`No candidates loaded from ${candidatesCsv}`);
        }
        if (this.booths.length === 0) {
            throw new Error(`No booths loaded from ${boothsCsv}`);
        }

        // Optional overrides from benchconfig
        this.encOneHex      = roundArguments.encOneHex      || '3';        // Enc(1) placeholder
        this.attestationSig = roundArguments.attestationSig || 'att-ok';   // non-empty for VerifyAttestation=true
        this.serialPrefix   = roundArguments.serialPrefix   || 'SERIAL';
        this.defaultConst   = roundArguments.constituencyID || 'C-001';    // matches your CSV, not "CONST-001"
    }

    pickRandom(arr) {
        const idx = Math.floor(Math.random() * arr.length);
        return arr[idx];
    }

    async submitTransaction() {
        const seq = this.txIndex++;

        // Randomly pick rows
        const voter     = this.pickRandom(this.voters);
        const candidate = this.pickRandom(this.candidates);
        const booth     = this.pickRandom(this.booths);

        // ---- Map CSV columns → RecordVote args ----
        //
        // RecordVote(
        //   serial, constituencyID, candidateID,
        //   encOneHex, receiptSalt, epoch, attestationSig,
        //   boothID, deviceID, deviceKeyFP,
        //   bioAlg, bioNonceB64, bioCipherB64, bioTagHex,
        // )

        // serial: use hashed voter id; fall back to synthetic if missing
        let serial = voter.voter_id_star;
        if (!serial) {
            serial = `${this.serialPrefix}-${this.workerIndex}-${seq}`;
        }

        // constituencyID: prefer booth.constituency_id, else candidate/voter, else default
        let constituencyID =
            booth.constituency_id ||
            candidate.constituency_id ||
            voter.constituency_id ||
            this.defaultConst;

        // candidateID from candidates.csv
        const candidateID = candidate.candidate_id || 'cand-000001';

        // receiptSalt: derive from serial (bench-safe)
        const receiptSalt = `salt-${serial}`;

        // epoch: simple increasing counter per worker
        const epoch = (seq + 1).toString();

        // booth/device data from booths.csv
        const boothID     = booth.booth_id               || 'B-0001';
        const deviceID    = booth.device_id              || 'D0010000000';
        const deviceKeyFP = booth.device_key_fingerprint || 'deadbeef';

        const args = [
            serial,
            constituencyID,
            candidateID,
            this.encOneHex,
            receiptSalt,
            epoch,
            this.attestationSig,
            boothID,
            deviceID,
            deviceKeyFP,
            '',  // bioAlg
            '',  // bioNonceB64
            '',  // bioCipherB64
            ''   // bioTagHex
        ];

        const request = {
            contractId: 'accumvote3',
            contractFunction: 'RecordVote',
            invokerIdentity: 'eci-admin',
            contractArguments: args,
            readOnly: false
        };

        await this.sutAdapter.sendRequests(request);
    }

    async cleanupWorkloadModule() {
        // nothing to do
    }
}

function createWorkloadModule() {
    return new RecordVoteWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
