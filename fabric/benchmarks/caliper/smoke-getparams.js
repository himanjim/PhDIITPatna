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

    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
    }

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

    async cleanupWorkloadModule() {
        // Nothing to cleanup
    }
}

function createWorkloadModule() {
    return new GetParamsWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;
