	'use strict';

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const { parse } = require('csv-parse/sync');
const {
  Gateway,
  Wallets,
  DefaultEventHandlerStrategies,
} = require('fabric-network');

// Record script start time (once, on load)
const SCRIPT_START = new Date();
console.log(`Script start: ${SCRIPT_START.toISOString()}`);

// --------- Fabric / file config ---------

const CHANNEL_NAME = 'statechan-01';
const CHAINCODE_NAME = 'accumvote3';

// BFT connection profile (Opp + Civil peers)
const CCP_PATH =
  process.env.CCP_PATH ||
  '/home/ubuntu/fab-election-bench/stacks/stack-v31-bft/crypto/connection-eci-bft_2of3.yaml';

// OppMSP admin MSP for BFT stack
const MSP_ROOT =
  '/home/ubuntu/fab-election-bench/stacks/stack-v31-bft/crypto/peerOrganizations/opp.bench.local/users/Admin@opp.bench.local/msp';


// Same CSVs Caliper uses
const DATA_ROOT = '/home/ubuntu/hyperledger/caliper/data';
const VOTERS_CSV = path.join(DATA_ROOT, 'voters.csv');
const CANDIDATES_CSV = path.join(DATA_ROOT, 'candidates.csv');
const BOOTHS_CSV = path.join(DATA_ROOT, 'booths.csv');

// Hard cap on in-flight txs (can be overridden via env)
const MAX_CONCURRENCY = parseInt(
  process.env.MAX_CONCURRENCY || '1024',
  10
);

// RecordVote arg defaults (mirrors recordvote.js)
const ENC_ONE_HEX = '3';
const ATTESTATION_SIG = 'att-ok';
const SERIAL_PREFIX = 'SERIAL';
const DEFAULT_CONST = 'C-001';

// --------- Helpers ---------

function loadCsv(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`CSV not found: ${filePath}`);
  }
  const content = fs.readFileSync(filePath, 'utf8');
  const records = parse(content, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  });
  if (!records.length) {
    throw new Error(`No rows in CSV: ${filePath}`);
  }
  return records;
}

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function percentile(latenciesMs, p) {
  if (!latenciesMs.length) return 0.0;
  const s = [...latenciesMs].sort((a, b) => a - b);
  const k = Math.round((p / 100) * (s.length - 1));
  return s[k];
}

// Map CSV rows → RecordVote args
// RecordVote(
//   serial, constituencyID, candidateID,
//   encOneHex, receiptSalt, epoch, attestationSig,
//   boothID, deviceID, deviceKeyFP,
//   bioAlg, bioNonceB64, bioCipherB64, bioTagHex
// )
function buildRecordVoteArgs(voter, candidate, booth, workerId, seq) {
  const serial =
    voter.voter_id_star && voter.voter_id_star.length
      ? voter.voter_id_star
      : `${SERIAL_PREFIX}-${workerId}-${seq}`;

  const constituencyId =
    booth.constituency_id ||
    candidate.constituency_id ||
    voter.constituency_id ||
    DEFAULT_CONST;

  const candidateId = candidate.candidate_id || 'cand-000001';
  const receiptSalt = `salt-${serial}`;
  const epoch = String(seq + 1);

  const boothId = booth.booth_id || 'B-0001';
  const deviceId = booth.device_id || 'D0010000000';
  const deviceKeyFP = booth.device_key_fingerprint || 'deadbeef';

  const args = [
    serial,
    constituencyId,
    candidateId,
    ENC_ONE_HEX,
    receiptSalt,
    epoch,
    ATTESTATION_SIG,
    boothId,
    deviceId,
    deviceKeyFP,
    '', // bioAlg
    '', // bioNonceB64
    '', // bioCipherB64
    '', // bioTagHex
  ];

  if (args.length !== 14) {
    throw new Error(`RecordVote arg length mismatch: got ${args.length}`);
  }
  return args;
}

// One tx: submit RecordVote and measure end-to-end latency
async function invokeRecordVote(contract, voters, candidates, booths, workerId, seq) {
  const voter = pickRandom(voters);
  const cand = pickRandom(candidates);
  const booth = pickRandom(booths);
  const args = buildRecordVoteArgs(voter, cand, booth, workerId, seq);

  const t0 = process.hrtime.bigint();
  await contract.submitTransaction('RecordVote', ...args);
  const t1 = process.hrtime.bigint();

  return Number(t1 - t0) / 1e6; // ms
}

// Run one round (Caliper-style): bounded concurrency, targetLoad → concurrency
async function runRound(label, txCount, targetLoad, contract, voters, candidates, booths) {
  const concurrency = Math.min(
    Math.max(targetLoad, 1),
    MAX_CONCURRENCY
  );

  console.log(
    `\n=== Round ${label}: tx=${txCount}, targetLoad=${targetLoad}, concurrency=${concurrency} ===`
  );

  let success = 0;
  let fail = 0;
  const latencies = [];

  let inFlight = 0;
  let submitted = 0;

  const start = process.hrtime.bigint();

  return new Promise((resolve) => {
    const launchMore = () => {
      // Fill the concurrency window
      while (inFlight < concurrency && submitted < txCount) {
        const seq = submitted++;
        const workerId = seq % concurrency;
        inFlight++;

        invokeRecordVote(contract, voters, candidates, booths, workerId, seq)
          .then((elapsedMs) => {
            latencies.push(elapsedMs);
            success++;
          })
          .catch((err) => {
            const msg = err && err.message ? err.message : String(err);
            latencies.push(0);
            fail++;
            // For max TPS, log only first error per round
            if (fail === 1) {
              console.error(`[ERR] first failure at tx#${seq}: ${msg}`);
            }
          })
          .finally(() => {
            inFlight--;
            if (success + fail === txCount) {
              const end = process.hrtime.bigint();
              const durSec = Number(end - start) / 1e9;
              const tps = durSec > 0 ? success / durSec : 0;
              const avg =
                latencies.length > 0
                  ? latencies.reduce((a, b) => a + b, 0) /
                    latencies.length
                  : 0;
              const p95 = percentile(latencies, 95);
              const p99 = percentile(latencies, 99);

              console.log(
                `Round ${label} done in ${durSec.toFixed(
                  2
                )}s | succ=${success}, fail=${fail}, TPS=${tps.toFixed(
                  2
                )}, avg=${avg.toFixed(1)}ms, p95=${p95.toFixed(
                  1
                )}ms, p99=${p99.toFixed(1)}ms`
              );

              resolve({
                label,
                txCount,
                targetLoad,
                concurrency,
                duration_s: durSec,
                success,
                fail,
                tps,
                avg_latency_ms: avg,
                p95_ms: p95,
                p99_ms: p99,
              });
            } else {
              // Not done yet, keep feeding
              launchMore();
            }
          });
      }
    };

    // Kick off initial batch
    launchMore();
  }).catch((err) => {
  const msg = err && err.message ? err.message : String(err);
  latencies.push(0);
  fail++;
  if (fail === 1) {
    console.error(`[ERR] first failure at tx#${seq}: ${msg}`);
  }
})
;
}

// --------- Main ---------

async function main() {
  // 1) Load CSVs (once)
  const voters = loadCsv(VOTERS_CSV);
  const candidates = loadCsv(CANDIDATES_CSV);
  const booths = loadCsv(BOOTHS_CSV);

  console.log(
    `Loaded voters=${voters.length}, candidates=${candidates.length}, booths=${booths.length}`
  );

  // 2) Load connection profile
  const ccpRaw = fs.readFileSync(CCP_PATH, 'utf8');
  const ccp = yaml.load(ccpRaw);

  // 3) Build wallet and identity from local MSP
  const wallet = await Wallets.newInMemoryWallet();

  const signcertsDir = path.join(MSP_ROOT, 'signcerts');
  const keystoreDir = path.join(MSP_ROOT, 'keystore');
  const certFile = fs
    .readdirSync(signcertsDir)
    .find((f) => f.endsWith('.pem'));
  const keyFile = fs.readdirSync(keystoreDir)[0];

  if (!certFile || !keyFile) {
    throw new Error('Could not find admin cert or key in MSP directories');
  }

  const cert = fs.readFileSync(path.join(signcertsDir, certFile), 'utf8');
  const key = fs.readFileSync(path.join(keystoreDir, keyFile), 'utf8');

	await wallet.put('opp-admin', {
	  credentials: {
		certificate: cert,
		privateKey: key,
	  },
	  mspId: 'OppMSP',
	  type: 'X.509',
	});

  // 4) Connect gateway
  //  - discovery disabled (we trust CCP peers)
  //  - event strategy ANY-for-TX (commit as soon as any peer reports it)
	const gateway = new Gateway();
	await gateway.connect(ccp, {
	  wallet,
	  identity: 'opp-admin',
	  // you can keep discovery disabled; the CCP now lists Opp + Civil as peers
	  discovery: { enabled: false, asLocalhost: false },
	  eventHandlerOptions: {
		strategy: DefaultEventHandlerStrategies.NETWORK_SCOPE_ANYFORTX,
		commitTimeout: 60,
	  },
	});


  const network = await gateway.getNetwork(CHANNEL_NAME);
  const contract = network.getContract(CHAINCODE_NAME);

  // 5) Decide rounds:
  //    - If TX is set in the environment, override YAML and run a single round.
  //    - Otherwise, use recordvote-steps.yaml if present.
  //    - If neither, fall back to a default single round.
  const stepsPath = path.resolve(process.cwd(), 'recordvote-steps.yaml');
  let rounds = [];

  const envTxRaw = process.env.TX;
  const envLoadRaw = process.env.LOAD;

  if (envTxRaw) {
    // Explicit override: one env-configured round
    const tx = parseInt(envTxRaw, 10);
    const load = parseInt(envLoadRaw || '500', 10);
    rounds = [
      {
        label: `env_tx${tx}_load${load}`,
        txCount: tx,
        targetLoad: load,
      },
    ];
  } else if (fs.existsSync(stepsPath)) {
    // YAML suite
    const stepsCfg = yaml.load(fs.readFileSync(stepsPath, 'utf8'));
    const yamlRounds = (stepsCfg.test && stepsCfg.test.rounds) || [];
    rounds = yamlRounds.map((r) => ({
      label: r.label || 'round',
      txCount: parseInt(r.txNumber, 10),
      targetLoad: parseInt(
        r.rateControl?.opts?.transactionLoad ?? '1',
        10
      ),
    }));
  } else {
    // Default single round if nothing else specified
    const tx = 10000;
    const load = 500;
    rounds = [
      {
        label: `default_tx${tx}_load${load}`,
        txCount: tx,
        targetLoad: load,
      },
    ];
  }

  // 6) Execute rounds
  const results = [];
  for (const r of rounds) {
    if (!r.txCount || r.txCount <= 0) continue;
    const res = await runRound(
      r.label,
      r.txCount,
      r.targetLoad,
      contract,
      voters,
      candidates,
      booths
    );
    results.push(res);
  }

  console.log('\n=== Summary ===');
  for (const r of results) {
    console.log(
      `${r.label}: tx=${r.txCount}, load=${r.targetLoad}, ` +
        `conc=${r.concurrency}, TPS=${r.tps.toFixed(2)}, ` +
        `avg=${r.avg_latency_ms.toFixed(1)}ms, p95=${r.p95_ms.toFixed(
          1
        )}ms, p99=${r.p99_ms.toFixed(1)}ms, succ=${r.success}, fail=${r.fail}`
    );
  }
  
    // Record script end time and total wall-clock duration
  const scriptEnd = new Date();
  const elapsedSec = (scriptEnd - SCRIPT_START) / 1000;
  console.log(`Script end:   ${scriptEnd.toISOString()}`);
  console.log(`Script elapsed: ${elapsedSec.toFixed(2)}s`);

  gateway.disconnect();
}

main().catch((err) => {
  console.error('FATAL:', err);
  process.exit(1);
});

