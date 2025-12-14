'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

class PingWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }

    async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
    }

    async submitTransaction() {
        const request = {
            contractId: 'accumvote2',
            contractFunction: 'Ping',
            // 🔴 CHANGE THIS LINE:
            invokerIdentity: 'eci-admin',   // was 'eci-admin'
            contractArguments: [],
            readOnly: false
        };

        await this.sutAdapter.sendRequests(request);
    }

    async cleanupWorkloadModule() {}
}

function createWorkloadModule() {
    return new PingWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;

