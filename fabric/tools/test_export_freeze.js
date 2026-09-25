/**
 * test_export_freeze.js - tests for the pack exporter.
 *
 * Run with:  node --test fabric/tools/
 *            node --test fabric/tools/test_export_freeze.js
 *
 * The tests cover the three things the exporter is trusted to get right: the
 * canonical JSON encoding that the freeze commitment is computed over, the
 * shape and ordering of the freeze list S, and the manifest that binds the
 * files of the pack together. The last test runs the exporter over the fixture
 * that the contract itself produced (fixtures/snapshot.json), so a change in
 * the contract's export format breaks a test here rather than in a deployment.
 */
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ef = require('./export_freeze.js');

const FIXTURES = path.join(__dirname, 'fixtures');

function tmpdir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'freeze-pack-'));
}

test('canonical JSON sorts keys and escapes control characters', () => {
  assert.strictEqual(ef.jcs({ b: 1, a: 2 }), '{"a":2,"b":1}');
  assert.strictEqual(ef.jcs({ 'B': 1, 'a': 1, 'A': 1 }), '{"A":1,"B":1,"a":1}');
  assert.strictEqual(ef.jcs(['x', 'y']), '["x","y"]');
  assert.strictEqual(ef.jcs({ s: 'a"b\\c\nd\te' }), '{"s":"a\\"b\\\\c\\nd\\te"}');
  assert.strictEqual(ef.jcs({ s: '\u0001' }), '{"s":"\\u0001"}');
  assert.strictEqual(ef.jcs({ t: true, f: false, n: null }), '{"f":false,"n":null,"t":true}');
});

test('the freeze list keeps the documented tuple shape and byte ordering', () => {
  const S = ef.buildS([
    { serial: 'S-002', hC: 'bb', txID: 't2', epoch: 'E1', castTime: 'T2', status: 'current' },
    { serial: 'S-0010', hC: 'cc', txID: 't10', epoch: 'E1', castTime: 'T10', status: 'invalid', reason: 'voter-roll' },
    { serial: 'S-001', hC: 'aa', txID: 't1', epoch: 'E1', castTime: 'T1', status: 'current' },
  ]);
  assert.deepStrictEqual(S.map((r) => r.serial), ['S-001', 'S-0010', 'S-002']);
  assert.deepStrictEqual(Object.keys(S[0]), ['serial', 'hC', 'txID', 'epoch', 'castTime', 'status', 'reason']);
  assert.strictEqual(S[0].reason, '', 'a missing reason is written as an empty string, not dropped');
});

test('a ballot list with a repeated serial is refused', () => {
  assert.throws(
    () => ef.buildS([{ serial: 'S-1', hC: 'aa' }, { serial: 'S-1', hC: 'bb' }]),
    /duplicate serial/
  );
});

test('HR is stable for a fixed list and changes when any field changes', () => {
  const rows = [{ serial: 'S-001', hC: 'aa', txID: 't1', epoch: 'E1', castTime: 'T1', status: 'current' }];
  const hr = ef.hrFromS(ef.buildS(rows));
  assert.strictEqual(hr, '52c7c51d1f50531a30edacee8243f74c2ff7e67c060862b4afad1edc2216092a');
  const moved = ef.hrFromS(ef.buildS([Object.assign({}, rows[0], { status: 'invalid' })]));
  assert.notStrictEqual(hr, moved);
});

test('the manifest lists a digest for every file, and the digests match', () => {
  const pack = ef.buildPack({
    ballots: [{ serial: 'S-1', hC: 'aa', encOneHex: '0f', status: 'current', txID: 't1' }],
    openings: [],
    params: { AUDIT_ONE_IN: 20 },
    publicKey: { n: 'ab', g: 'ac' },
    options: ['option-0'],
    meta: { generatedAt: '2026-01-01T00:00:00Z' },
  });
  const crypto = require('crypto');
  for (const [name, entry] of Object.entries(pack.manifest.files)) {
    const digest = crypto.createHash('sha256').update(pack.files[name], 'utf8').digest('hex');
    assert.strictEqual(digest, entry.sha256, name + ' digest');
  }
  assert.ok(!('manifest.json' in pack.manifest.files), 'the manifest does not list itself');
  assert.strictEqual(pack.manifest.counts.current, 1);
  assert.strictEqual(pack.manifest.HR, ef.hrFromS(pack.S));
});

test('audit openings are sorted, deduplicated, and stripped of booth, device and time', () => {
  const pack = ef.buildPack({
    ballots: [{ serial: 'S-1', hC: 'aa', encOneHex: '0f', status: 'current' }],
    openings: [
      { hC: 'ff', constituencyID: 'C-001', optionIndex: 1, randomnessHex: '07', boothID: 'B-0001', deviceID: 'D-1', openedAt: '2026-01-01T10:00:00Z' },
      { hC: '0a', constituencyID: 'C-001', optionIndex: 0, randomnessHex: '05', boothID: 'B-0002', deviceID: 'D-2' },
    ],
    params: {},
    publicKey: {},
  });
  const openings = JSON.parse(pack.files['openings.json']);
  assert.deepStrictEqual(openings.map((o) => o.hC), ['0a', 'ff']);
  for (const o of openings) {
    assert.ok(!('boothID' in o), 'booth identifiers must not reach the public pack');
    assert.ok(!('deviceID' in o), 'device identifiers must not reach the public pack');
    assert.ok(!('openedAt' in o), 'opening times must not reach the public pack');
  }
  assert.throws(
    () => ef.buildPack({ ballots: [{ serial: 'S-1', hC: 'aa' }], openings: [{ hC: 'aa', optionIndex: 0, randomnessHex: '01' }, { hC: 'aa', optionIndex: 0, randomnessHex: '02' }] }),
    /duplicate audit opening/
  );
});

test('the exporter runs over the contract-produced fixture and writes a complete pack', () => {
  const snapshot = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'snapshot.json'), 'utf8'));
  const pack = ef.buildPack({
    ballots: snapshot.ballots,
    openings: snapshot.openings,
    querylog: snapshot.querylog,
    params: snapshot.params,
    publicKey: snapshot.publicKey,
    options: snapshot.options,
    tally: snapshot.tally,
    results: snapshot.results,
    meta: snapshot.meta,
  });
  const dir = tmpdir();
  const written = ef.writePack(dir, pack);
  assert.ok(written.includes('S.json') && written.includes('manifest.json') && written.includes('ballots.json'));

  // The contract exported eleven ballots, ten of them current, and three openings.
  assert.strictEqual(pack.manifest.counts.ballots, 11);
  assert.strictEqual(pack.manifest.counts.current, 10);
  assert.strictEqual(pack.manifest.counts.openings, 3);

  // Every current ballot carries a ciphertext that hashes to its commitment.
  const crypto = require('crypto');
  for (const b of JSON.parse(pack.files['ballots.json'])) {
    if (b.status !== 'current') continue;
    assert.strictEqual(crypto.createHash('sha256').update(b.encOneHex, 'utf8').digest('hex'), b.hC, b.serial);
  }

  // Running twice over the same input gives byte-identical files.
  const again = ef.buildPack({
    ballots: snapshot.ballots, openings: snapshot.openings, querylog: snapshot.querylog,
    params: snapshot.params, publicKey: snapshot.publicKey, options: snapshot.options,
    tally: snapshot.tally, results: snapshot.results, meta: snapshot.meta,
  });
  assert.strictEqual(again.files['S.json'], pack.files['S.json']);
  assert.strictEqual(again.HR, pack.HR);
  fs.rmSync(dir, { recursive: true, force: true });
});

test('the command line refuses an incomplete invocation', async () => {
  await assert.rejects(() => ef.main(['--out', tmpdir()]), /--snapshot/);
});
