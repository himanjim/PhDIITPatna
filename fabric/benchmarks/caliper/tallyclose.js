'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class TallyCloseWorkload extends WorkloadModuleBase {

    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);

        this.mode          = (roundArguments.mode || 'tally').toLowerCase();
        this.constituency  = roundArguments.constituencyID || 'C-001';
        this.roundID       = roundArguments.roundID       || 'R1';
        this.resultsJSON   = roundArguments.resultsJSON   || '{"CAND-001": 1}';
        this.bundleHash    = roundArguments.bundleHash    || 'deadwood';
    }

    async submitTransaction() {
        let request;

        if (this.mode === 'close') {
            request = {
                contractId: 'accumvote2',
                contractFunction: 'ClosePoll',
                invokerIdentity: 'eci-admin',
                contractArguments: [this.constituency],
                readOnly: false
            };
        } else if (this.mode === 'tally') {
	    request = {
		contractId: 'accumvote2',
		contractFunction: 'TallyPrepare',
		invokerIdentity: 'eci-admin',
		contractArguments: [this.constituency],
		readOnly: true           // ← THIS IS THE IMPORTANT CHANGE
	    };
	}
	 else if (this.mode === 'publish') {
            request = {
                contractId: 'accumvote2',
                contractFunction: 'PublishResults',
                invokerIdentity: 'eci-admin',
                contractArguments: [
                    this.constituency,
                    this.roundID,
                    this.resultsJSON,
                    this.bundleHash
                ],
                readOnly: false
            };
        } else {
            throw new Error(`Unknown mode: ${this.mode}`);
        }

        await this.sutAdapter.sendRequests(request);
    }

    async cleanupWorkloadModule() {}
}

function createWorkloadModule() {
    return new TallyCloseWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
