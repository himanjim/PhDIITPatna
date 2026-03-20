// ping.js defines a minimal Caliper workload module that invokes the Ping function on the target chaincode. Its purpose is to verify basic client-to-network reachability, identity resolution, and transaction submission plumbing before more expensive benchmark rounds such as RecordVote or tally workflows are attempted.

'use strict';

const { WorkloadModuleBase } = require('@hyperledger/caliper-core');

// PingWorkload is intentionally small. It inherits the standard Caliper workload lifecycle but contributes only a single transaction shape, making it useful as a smoke-test harness and as a low-noise baseline when diagnosing client, gateway, or chaincode connectivity problems.
class PingWorkload extends WorkloadModuleBase {
    constructor() {
        super();
    }
    
    // This lifecycle hook delegates initialisation to the Caliper base class and does not add workload-specific state. It is     retained explicitly so that future variants can attach round arguments or worker-local configuration without changing the     surrounding module structure.
        async initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext) {
        await super.initializeWorkloadModule(workerIndex, totalWorkers, roundIndex, roundArguments, sutAdapter, sutContext);
    }

    // submitTransaction builds the fixed request envelope used in each benchmark iteration. The request targets the Ping         function on the configured contract and submits it through one named invoker identity, allowing the surrounding benchmark     to isolate transport and endorsement-path overhead from application-level argument generation.
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

    // No workload-local cleanup is required because this module does not allocate temporary resources or maintain per-worker     state beyond what Caliper manages itself.
    async cleanupWorkloadModule() {}
}

// Caliper expects each workload file to export a factory function that returns a fresh workload-module instance for the worker process that imports it.
function createWorkloadModule() {
    return new PingWorkload();
}

module.exports.createWorkloadModule = createWorkloadModule;

