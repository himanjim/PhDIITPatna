/**
 * Caliper workload module for the administrative phases that follow vote
 * recording in the accumvote flow. Depending on the selected mode, the
 * module issues one of three contract calls: poll closure, tally
 * preparation, or result publication. The purpose of the module is to
 * let benchmark rounds exercise these post-voting transitions through a
 * single configurable workload implementation.
 */

'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');
/**
 * Workload implementation for administrative chaincode functions that are
 * driven by round-level arguments rather than by per-transaction random
 * data generation.
 */
class TallyCloseWorkload extends WorkloadModuleBase {

    /**
     * Read the round arguments supplied by the Caliper benchmark file and
     * cache the parameters needed to construct the correct contract
     * invocation for the selected mode.
     */
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);

        this.mode          = (roundArguments.mode || 'tally').toLowerCase();
        this.constituency  = roundArguments.constituencyID || 'C-001';
        this.roundID       = roundArguments.roundID       || 'R1';
        this.resultsJSON   = roundArguments.resultsJSON   || '{"CAND-001": 1}';
        this.bundleHash    = roundArguments.bundleHash    || 'deadwood';
    }

    /**
     * Build and submit exactly one administrative transaction or
     * read-only evaluation, depending on the configured workload mode.
     */
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
		readOnly: true           
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

/**
 * Caliper entry point for creating one workload instance per worker.
 */
function createWorkloadModule() {
    return new TallyCloseWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
