#!/usr/bin/env node
/**
 * export_freeze.js - build the public verification pack for one constituency at
 * the freeze point.
 *
 * The pack is the input of tools/verify_public.py and of any third-party
 * verifier. It contains only public material: the frozen ballot list, the
 * published ballot ciphertexts, the audit log of opened ballots, the query log,
 * the published parameters and the election public key. Nothing in the pack
 * comes from a private data collection.
 *
 * Two sources are supported:
 *
 *   1. Files. The JSON returned by the chaincode methods ExportPublicBallots,
 *      ExportAuditOpenings, GetParams and GetJointPublicKey is read from disk.
 *      This is the mode used by the tests and by anyone reproducing a published
 *      pack offline.
 *
 *   2. A Fabric gateway. The same four methods are evaluated against a peer.
 *      This path needs @hyperledger/fabric-gateway and @grpc/grpc-js, which are
 *      loaded only when --gateway is given, so the tool runs without them.
 *
 * The freeze commitment is computed exactly as the key-ceremony component
 * defines it:
 *
 *      S  = [(serial, hC, txID, epoch, castTime, status, reason)] sorted by
 *           serial in ascending byte order
 *      HR = SHA256("HR_LIST_V1" || JCS(S))
 *
 * where JCS is RFC 8785 canonical JSON. Writing HR here rather than trusting a
 * value supplied by the election administrator is deliberate: the verifier
 * recomputes it again from S, and the two must agree with the value anchored on
 * the ledger.
 *
 * Usage
 *   node export_freeze.js --out PACKDIR --ballots b.json --openings a.json \
 *        --params p.json --pubkey k.json [--options o.json] [--querylog q.json] \
 *        [--eid EID] [--constituency C-001] [--block 1234] [--txid abcd]
 *
 *   node export_freeze.js --out PACKDIR --snapshot snapshot.json
 *
 *   node export_freeze.js --out PACKDIR --gateway --profile gw.json \
 *        --constituency C-001 --state UP
 *
 * Exit status is 0 on success and 1 on any error.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const TOOL = 'export_freeze.js';
const VERSION = '1.0.0';
const HR_DOMAIN = 'HR_LIST_V1';
const S_FIELDS = ['serial', 'hC', 'txID', 'epoch', 'castTime', 'status', 'reason'];

/* ------------------------------------------------------------------ *
 * RFC 8785 canonical JSON (the subset the pack needs: objects, arrays,
 * strings, integers, booleans and null). Keys are sorted by UTF-16 code
 * unit, there is no insignificant whitespace, and strings use the short
 * escapes where they exist.
 * ------------------------------------------------------------------ */
function jcsString(s) {
  let out = '"';
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === '"') out += '\\"';
    else if (ch === '\\') out += '\\\\';
    else if (c === 0x08) out += '\\b';
    else if (c === 0x09) out += '\\t';
    else if (c === 0x0a) out += '\\n';
    else if (c === 0x0c) out += '\\f';
    else if (c === 0x0d) out += '\\r';
    else if (c < 0x20) out += '\\u' + c.toString(16).padStart(4, '0');
    else out += ch;
  }
  return out + '"';
}

function jcs(value) {
  if (value === null) return 'null';
  const t = typeof value;
  if (t === 'boolean') return value ? 'true' : 'false';
  if (t === 'number') {
    if (!Number.isFinite(value)) throw new Error('JCS: non-finite number');
    return Number.isInteger(value) ? String(value) : String(value);
  }
  if (t === 'string') return jcsString(value);
  if (Array.isArray(value)) return '[' + value.map(jcs).join(',') + ']';
  if (t === 'object') {
    const keys = Object.keys(value).filter((k) => value[k] !== undefined).sort(compareUtf16);
    return '{' + keys.map((k) => jcsString(k) + ':' + jcs(value[k])).join(',') + '}';
  }
  throw new Error('JCS: unsupported value of type ' + t);
}

function compareUtf16(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

function sha256Hex(data) {
  return crypto.createHash('sha256').update(data, 'utf8').digest('hex');
}

/* ------------------------------------------------------------------ *
 * Pack construction
 * ------------------------------------------------------------------ */

/**
 * buildS turns the exported public ballots into the freeze list S: one tuple per
 * serial, fields in the order the key-ceremony component fixes, sorted by serial
 * in ascending byte order. Absent optional fields are written as empty strings
 * so that every tuple has the same shape.
 */
function buildS(ballots) {
  const rows = ballots.map((b) => {
    const row = {};
    for (const f of S_FIELDS) row[f] = b[f] === undefined || b[f] === null ? '' : String(b[f]);
    if (!row.serial) throw new Error('ballot row without a serial: ' + JSON.stringify(b));
    if (!row.hC) throw new Error('ballot row without a commitment hC: ' + row.serial);
    return row;
  });
  rows.sort((x, y) => compareBytes(x.serial, y.serial));
  for (let i = 1; i < rows.length; i++) {
    if (rows[i].serial === rows[i - 1].serial) throw new Error('duplicate serial in ballot export: ' + rows[i].serial);
  }
  return rows;
}

function compareBytes(a, b) {
  const ba = Buffer.from(a, 'utf8');
  const bb = Buffer.from(b, 'utf8');
  return Buffer.compare(ba, bb);
}

function hrFromS(S) {
  return sha256Hex(HR_DOMAIN + jcs(S));
}

function normaliseBallots(ballots) {
  const out = ballots.map((b) => ({
    serial: String(b.serial),
    hC: String(b.hC || '').toLowerCase(),
    encOneHex: b.encOneHex === undefined ? '' : String(b.encOneHex),
    status: b.status === undefined ? '' : String(b.status),
    epoch: b.epoch === undefined ? '' : String(b.epoch),
    castTime: b.castTime === undefined ? '' : String(b.castTime),
    txID: b.txID === undefined ? '' : String(b.txID),
    reason: b.reason === undefined ? '' : String(b.reason),
  }));
  out.sort((x, y) => compareBytes(x.serial, y.serial));
  return out;
}

/**
 * normaliseOpenings keeps only the four fields a third party needs to repeat the
 * kiosk's check. The booth, the device and the time of an opening are dropped
 * on purpose: an opening shows the option the voter had just chosen, so a booth
 * and a time would come close to naming that voter. Those fields stay in the
 * restricted audit record that Tier B uses for fault attribution (dashboard
 * component, Section 3.10A).
 */
function normaliseOpenings(openings) {
  const out = openings.map((o) => ({
    hC: String(o.hC || '').toLowerCase(),
    constituencyID: o.constituencyID === undefined ? '' : String(o.constituencyID),
    optionIndex: Number(o.optionIndex),
    randomnessHex: String(o.randomnessHex || '').toLowerCase(),
  }));
  for (const o of out) {
    if (!o.hC) throw new Error('audit opening without a commitment');
    if (!Number.isInteger(o.optionIndex) || o.optionIndex < 0) throw new Error('audit opening with a bad option index for ' + o.hC);
    if (!o.randomnessHex) throw new Error('audit opening without randomness for ' + o.hC);
  }
  out.sort((x, y) => compareBytes(x.hC, y.hC));
  for (let i = 1; i < out.length; i++) {
    if (out[i].hC === out[i - 1].hC) throw new Error('duplicate audit opening for ' + out[i].hC);
  }
  return out;
}


function normaliseQueryLog(entries) {
  const out = (entries || []).map((q) => (typeof q === 'string' ? { hC: q.toLowerCase() } : { hC: String(q.hC || '').toLowerCase() }));
  for (const q of out) if (!q.hC) throw new Error('query log entry without a commitment');
  out.sort((x, y) => compareBytes(x.hC, y.hC));
  return out;
}

/**
 * buildPack assembles the files of the pack and the manifest that binds them.
 * It returns plain strings so that the caller can write them, hash them, or
 * compare them in a test without touching the file system.
 */
function buildPack(input) {
  const ballots = normaliseBallots(input.ballots || []);
  const openings = normaliseOpenings(input.openings || []);
  const querylog = normaliseQueryLog(input.querylog);
  const S = buildS(ballots);
  const HR = hrFromS(S);

  const counts = {
    ballots: ballots.length,
    current: ballots.filter((b) => b.status === 'current').length,
    withCiphertext: ballots.filter((b) => b.encOneHex !== '').length,
    openings: openings.length,
    queries: querylog.length,
  };

  const files = {
    'S.json': jcs(S),
    'ballots.json': jcs(ballots),
    'openings.json': jcs(openings),
    'querylog.json': jcs(querylog),
    'params.json': jcs(input.params || {}),
    'publickey.json': jcs(input.publicKey || {}),
    'options.json': jcs(input.options || []),
  };
  if (input.tally !== undefined && input.tally !== null) files['tally.json'] = jcs(input.tally);
  if (input.results !== undefined && input.results !== null) files['results.json'] = jcs(input.results);

  const fileIndex = {};
  for (const name of Object.keys(files).sort(compareUtf16)) {
    fileIndex[name] = { sha256: sha256Hex(files[name]), bytes: Buffer.byteLength(files[name], 'utf8') };
  }

  const meta = input.meta || {};
  const manifest = {
    tool: TOOL,
    version: VERSION,
    generatedAt: meta.generatedAt || new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    eid: meta.eid || '',
    constituencyID: meta.constituencyID || '',
    freeze: { blockHeight: meta.blockHeight === undefined ? '' : String(meta.blockHeight), txID: meta.txID || '' },
    hrDomain: HR_DOMAIN,
    HR,
    counts,
    files: fileIndex,
  };
  files['manifest.json'] = jcs(manifest);
  return { files, manifest, S, HR };
}

function writePack(dir, pack) {
  fs.mkdirSync(dir, { recursive: true });
  for (const [name, body] of Object.entries(pack.files)) {
    fs.writeFileSync(path.join(dir, name), body + '\n', 'utf8');
  }
  return Object.keys(pack.files).sort(compareUtf16);
}

/* ------------------------------------------------------------------ *
 * Input handling
 * ------------------------------------------------------------------ */
function readJson(p) {
  const raw = fs.readFileSync(p, 'utf8');
  try {
    return JSON.parse(raw);
  } catch (e) {
    throw new Error('not valid JSON: ' + p + ' (' + e.message + ')');
  }
}

/**
 * Chaincode methods return JSON as a string. A caller who pipes their output
 * straight into a file therefore ends up with a quoted string rather than an
 * array, so both forms are accepted.
 */
function readJsonMaybeString(p) {
  const v = readJson(p);
  return typeof v === 'string' ? JSON.parse(v) : v;
}

function parseArgs(argv) {
  const opts = { out: 'freeze_pack' };
  const flags = new Set(['gateway', 'help']);
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) throw new Error('unexpected argument: ' + a);
    const key = a.slice(2);
    if (flags.has(key)) {
      opts[key] = true;
      continue;
    }
    const value = argv[++i];
    if (value === undefined) throw new Error('missing value for --' + key);
    opts[key] = value;
  }
  return opts;
}

function collectFromFiles(opts) {
  if (opts.snapshot) {
    const s = readJson(opts.snapshot);
    return {
      ballots: s.ballots || [],
      openings: s.openings || [],
      querylog: s.querylog || [],
      params: s.params || {},
      publicKey: s.publicKey || s.pubkey || {},
      options: s.options || s.optList || [],
      tally: s.tally,
      results: s.results,
      meta: s.meta || {},
    };
  }
  if (!opts.ballots) throw new Error('give --snapshot, or --ballots with --openings, --params and --pubkey, or --gateway');
  return {
    ballots: readJsonMaybeString(opts.ballots),
    openings: opts.openings ? readJsonMaybeString(opts.openings) : [],
    querylog: opts.querylog ? readJsonMaybeString(opts.querylog) : [],
    params: opts.params ? readJsonMaybeString(opts.params) : {},
    publicKey: opts.pubkey ? readJsonMaybeString(opts.pubkey) : {},
    options: opts.options ? readJsonMaybeString(opts.options) : [],
    tally: opts.tally ? readJsonMaybeString(opts.tally) : undefined,
    results: opts.results ? readJsonMaybeString(opts.results) : undefined,
    meta: {},
  };
}

/**
 * collectFromGateway evaluates the four read-only methods against a peer. The
 * gateway libraries are required lazily, so a verifier who only reads published
 * files never has to install them. This path is exercised in a deployment, not
 * by the offline tests, and that limitation is stated in the README.
 */
async function collectFromGateway(opts) {
  let grpc, gw, fsp;
  try {
    grpc = require('@grpc/grpc-js');
    gw = require('@hyperledger/fabric-gateway');
    fsp = require('fs').promises;
  } catch (e) {
    throw new Error('--gateway needs @grpc/grpc-js and @hyperledger/fabric-gateway to be installed: ' + e.message);
  }
  if (!opts.profile) throw new Error('--gateway needs --profile pointing at a gateway profile JSON');
  const p = readJson(opts.profile);
  const required = ['peerEndpoint', 'peerHostAlias', 'tlsRootCert', 'mspId', 'certPath', 'keyPath', 'channel', 'chaincode'];
  for (const k of required) if (!p[k]) throw new Error('gateway profile is missing "' + k + '"');

  const tls = grpc.credentials.createSsl(await fsp.readFile(p.tlsRootCert));
  const client = new grpc.Client(p.peerEndpoint, tls, { 'grpc.ssl_target_name_override': p.peerHostAlias });
  const identity = { mspId: p.mspId, credentials: await fsp.readFile(p.certPath) };
  const privateKey = gw.signers.newPrivateKeySigner(require('crypto').createPrivateKey(await fsp.readFile(p.keyPath)));
  const connection = gw.connect({ client, identity, signer: privateKey });
  try {
    const contract = connection.getNetwork(p.channel).getContract(p.chaincode);
    const text = async (fn, ...args) => new TextDecoder().decode(await contract.evaluateTransaction(fn, ...args));
    const ballots = JSON.parse(await text('ExportPublicBallots'));
    const openings = JSON.parse(await text('ExportAuditOpenings'));
    const params = JSON.parse(await text('GetParams'));
    const publicKey = JSON.parse(await text('GetJointPublicKey', opts.state || p.state || ''));
    let options = [];
    if (opts.constituency) {
      try {
        options = JSON.parse(await text('GetCandidateList', opts.constituency));
      } catch (e) {
        options = [];
      }
    }
    return { ballots, openings, querylog: [], params, publicKey, options, meta: {} };
  } finally {
    connection.close();
    client.close();
  }
}

const USAGE = `${TOOL} ${VERSION}
  node export_freeze.js --out PACKDIR --ballots b.json --openings a.json --params p.json --pubkey k.json
                        [--options o.json] [--querylog q.json] [--tally t.json] [--results r.json]
                        [--eid EID] [--constituency C-001] [--block N] [--txid TXID]
  node export_freeze.js --out PACKDIR --snapshot snapshot.json
  node export_freeze.js --out PACKDIR --gateway --profile gw.json --state UP --constituency C-001
`;

async function main(argv) {
  const opts = parseArgs(argv);
  if (opts.help) {
    process.stdout.write(USAGE);
    return 0;
  }
  const input = opts.gateway ? await collectFromGateway(opts) : collectFromFiles(opts);
  input.meta = Object.assign({}, input.meta, {
    eid: opts.eid || input.meta.eid,
    constituencyID: opts.constituency || input.meta.constituencyID,
    blockHeight: opts.block || input.meta.blockHeight,
    txID: opts.txid || input.meta.txID,
  });
  const pack = buildPack(input);
  const written = writePack(opts.out, pack);
  process.stdout.write(
    `wrote ${written.length} files to ${opts.out}\n` +
      `  ballots      ${pack.manifest.counts.ballots} (current ${pack.manifest.counts.current}, with ciphertext ${pack.manifest.counts.withCiphertext})\n` +
      `  openings     ${pack.manifest.counts.openings}\n` +
      `  query log    ${pack.manifest.counts.queries}\n` +
      `  HR           ${pack.HR}\n`
  );
  return 0;
}

if (require.main === module) {
  main(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((err) => {
      process.stderr.write(TOOL + ': ' + err.message + '\n');
      process.exit(1);
    });
}

module.exports = { jcs, sha256Hex, buildS, hrFromS, buildPack, writePack, normaliseBallots, normaliseOpenings, normaliseQueryLog, parseArgs, main, VERSION, HR_DOMAIN };
