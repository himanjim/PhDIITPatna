/**
 * Minimal Caliper workload module for repeated read-only evaluation of the
 * accumvote contract. The workload is intentionally simple so that it can
 * serve as a connectivity and configuration check with negligible
 * chaincode-side business logic.
 */
'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

/**
 * Simple workload that repeatedly calls GetParams on the accumvote contract.
 * This verifies Caliper <-> Fabric Gateway <-> chaincode connectivity.
 */
class GetParamsWorkload extends WorkloadModuleBase {

    constructor() {
        super();
    }
    
    /**
     * Delegate standard Caliper workload initialisation. This workload does
     * not require additional per-round state beyond the base setup.
     */
    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
    }

    /**
     * Submit one read-only GetParams evaluation request to the target
     * contract.
     */
    async submitTransaction() {
        const request = {
            contractId: 'accumvote',
            contractFunction: 'GetParams',
            invokerIdentity: 'eci-admin',
            contractArguments: [],
            readOnly: true
        };

        await this.sutAdapter.sendRequests(request);
    }

    /**
     * No workload-specific cleanup is required for this module.
     */
    async cleanupWorkloadModule() {
 
    }
}
/**
 * Caliper entry point for constructing one workload instance per worker.
 */
function createWorkloadModule() {
    return new GetParamsWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
