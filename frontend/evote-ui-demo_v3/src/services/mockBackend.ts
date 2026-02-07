/**
 * In-browser mock backend.
 * - Intercepts fetch() calls to /api/*
 * - Stores state in-memory (NOT persistent) to mimic server-side isolation
 * - Implements:
 *    /api/session/start, /api/session/end
 *    /api/ballot
 *    /api/liveness
 *    /api/vote/cast
 *    /api/verifier/enroll
 *    /api/receipt/verify
 *
 * The shapes align with the UI document:
 *  - Ballot publication (read-only) separate from casting (write)
 *  - Receipt payload: {v,E,s,hC,nonce,sealed}
 *  - Client B is kiosk-only by construction: verify endpoint requires device credential
 */

import type { BallotResp, CastReq, CastResp, EnrollReq, EnrollResp, LivenessReq, LivenessResp, SessionStartReq, SessionStartResp, VerifyReq, VerifyResp } from "./api";

type Session = {
  sessionId: string;
  sessionToken: string;
  mode: "remote" | "kiosk";
  voterId: string;
  constituencyId: string;
  ballotReadToken: string;
  castToken: string;
  expiresAt: number;
  livenessPassed: boolean;
};

type VoteRecord = {
  voterId: string;
  constituencyId: string;
  contestId: string;
  candidateId: string;
  txID: string;
  epoch: number;
  serial: string;
  hC: string;
  castTime: string;
  nonce: string;
  sealed: string;
  shortCode: string;
  qrPayload: string;
};

type Device = { deviceId: string; deviceToken: string; boothId: string; expiresAt: number };

// --- In-memory stores (server-side in a real deployment) ---
const SESSIONS = new Map<string, Session>();         // sessionId -> Session
const SESSION_BY_TOKEN = new Map<string, string>();  // token -> sessionId
const DEVICES = new Map<string, Device>();           // deviceId -> Device
const VOTES_BY_SERIAL = new Map<string, VoteRecord>();
const VOTES_BY_VOTER = new Map<string, VoteRecord>(); // last-vote-wins (overwrite)
const SHORT_TO_SERIAL = new Map<string, string>();

function now() { return Date.now(); }
function iso(ms: number) { return new Date(ms).toISOString(); }
function randHex(n = 16) {
  const a = new Uint8Array(n);
  crypto.getRandomValues(a);
  return [...a].map(x => x.toString(16).padStart(2, "0")).join("");
}
async function sha256Hex(s: string) {
  const b = new TextEncoder().encode(s);
  const h = await crypto.subtle.digest("SHA-256", b);
  return [...new Uint8Array(h)].map(x => x.toString(16).padStart(2, "0")).join("");
}
function base32Short(hex: string) {
  // Simple human-friendly code: take 10 chars from hex, group.
  const core = hex.slice(0, 10).toUpperCase();
  return `${core.slice(0,5)}-${core.slice(5)}`;
}

/** Minimal ballot generator per constituency. */
async function makeBallot(constituencyId: string): Promise<BallotResp> {
  const electionId = "ELECT-2026-DEMO";
  const contestId = `PC-${constituencyId}`;
  const candidates = [
    { id: "C1", name: "Candidate 1", party: "Party A" },
    { id: "C2", name: "Candidate 2", party: "Party B" },
    { id: "C3", name: "Candidate 3", party: "Party C" }
  ];
  // Digest is a hash over the serialized ballot definition.
  const digest = await sha256Hex(JSON.stringify({ electionId, constituencyId, contestId, candidates }));
  return { electionId, constituencyId, contestId, digest, candidates };
}

function json(body: any, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" }
  });
}

function unauthorized(msg = "unauthorized") {
  return new Response(msg, { status: 401 });
}

function forbidden(msg = "forbidden") {
  return new Response(msg, { status: 403 });
}

function notFound() {
  return new Response("not found", { status: 404 });
}

function getAuthToken(req: Request): string | null {
  const h = req.headers.get("authorization") || "";
  const m = h.match(/^Bearer\s+(.+)$/i);
  return m ? m[1] : null;
}

function getDeviceToken(req: Request): string | null {
  const h = req.headers.get("authorization") || "";
  const m = h.match(/^Device\s+(.+)$/i);
  return m ? m[1] : null;
}

async function handleStartSession(req: Request) {
  const body = (await req.json()) as SessionStartReq;

  // Minimal input validation.
  if (!body.voterId || !body.constituencyId || !body.mode) {
    return json({ error: "missing required fields" }, 400);
  }

  // Kiosk-only officer check (placeholder; replace with FIDO2/WebAuthn in real system).
  if (body.mode === "kiosk") {
    if (!body.officerPin || body.officerPin.length < 4) {
      return json({ error: "officerPin required for kiosk mode" }, 400);
    }
  }

  const sessionId = `S-${randHex(8)}`;
  const sessionToken = `st_${randHex(18)}`;
  const ballotReadToken = `br_${randHex(16)}`;
  const castToken = `ct_${randHex(16)}`;
  const ttlMs = 3 * 60 * 1000; // 3 minutes demo TTL
  const expiresAt = now() + ttlMs;

  const s: Session = {
    sessionId,
    sessionToken,
    mode: body.mode,
    voterId: body.voterId,
    constituencyId: body.constituencyId,
    ballotReadToken,
    castToken,
    expiresAt,
    livenessPassed: false
  };

  SESSIONS.set(sessionId, s);
  SESSION_BY_TOKEN.set(sessionToken, sessionId);

  const resp: SessionStartResp = {
    sessionId,
    sessionToken,
    expiresAt: iso(expiresAt),
    mode: body.mode,
    constituencyId: body.constituencyId,
    capabilities: { ballotReadToken, castToken },
    boothContext: body.mode === "kiosk" ? { boothId: "BOOTH-17", terminal: "A" } : undefined
  };
  return json(resp);
}

async function handleEndSession(req: Request) {
  const token = getAuthToken(req);
  if (!token) return unauthorized();
  const sid = SESSION_BY_TOKEN.get(token);
  if (!sid) return unauthorized();

  const body = await req.json().catch(() => ({}));
  if (body.sessionId && body.sessionId !== sid) return forbidden("session mismatch");

  // Invalidate session token and delete session state.
  SESSION_BY_TOKEN.delete(token);
  SESSIONS.delete(sid);
  return json({ ok: true });
}

async function handleBallot(req: Request, url: URL) {
  const token = getAuthToken(req);
  if (!token) return unauthorized();
  // In this demo, we accept any bearer token for ballot reads.
  const constituencyId = url.searchParams.get("constituencyId") || "";
  if (!constituencyId) return json({ error: "missing constituencyId" }, 400);
  return json(await makeBallot(constituencyId));
}

async function handleLiveness(req: Request) {
  const token = getAuthToken(req);
  if (!token) return unauthorized();
  const sid = SESSION_BY_TOKEN.get(token);
  if (!sid) return unauthorized();

  const session = SESSIONS.get(sid);
  if (!session) return unauthorized();
  if (session.expiresAt < now()) return json({ error: "session expired" }, 401);

  const body = (await req.json()) as LivenessReq;
  if (!body.sessionId || body.sessionId !== sid) return forbidden("session mismatch");

  // Dummy liveness: require at least 2 frames and basic entropy in payload length.
  const frames = body.stillsJpegB64 || [];
  const avgLen = frames.length ? frames.reduce((a, s) => a + s.length, 0) / frames.length : 0;

  const passed = frames.length >= 2 && avgLen > 8000;
  const score = passed ? 0.92 : 0.32;

  session.livenessPassed = passed;

  // Dummy dedup: if voter already has a vote recorded => REVOTE.
  const prev = VOTES_BY_VOTER.get(session.voterId);
  const dedup = prev ? { status: "REVOTE" as const, prev: { castTime: prev.castTime, boothId: "BOOTH-17", receiptShortCode: prev.shortCode } } : { status: "NEW" as const };

  const resp: LivenessResp = { passed, score, dedup };
  return json(resp);
}

async function handleCast(req: Request) {
  const token = getAuthToken(req);
  if (!token) return unauthorized();

  // Here we accept any castToken; in a real gateway, token must carry scope "cast".
  const body = (await req.json()) as CastReq;
  if (!body.sessionId || !body.contestId || !body.candidateId || !body.ballotDigest) {
    return json({ error: "missing required fields" }, 400);
  }

  const session = SESSIONS.get(body.sessionId);
  if (!session) return unauthorized();
  if (session.expiresAt < now()) return json({ error: "session expired" }, 401);
  if (!session.livenessPassed) return json({ error: "liveness required before casting" }, 403);

  // Build a receipt-like acknowledgement.
  const txID = `TX-${randHex(10)}`;
  const epoch = 1; // Demo epoch
  const serial = `SER-${randHex(8)}`;
  const castTime = iso(now());
  const nonce = randHex(12);

  // Commitment hash hC is a hash over (serial|candidateId|nonce|ballotDigest).
  const hC = await sha256Hex(`${serial}|${body.candidateId}|${nonce}|${body.ballotDigest}`);

  // "sealed" is opaque to Client A (designated verifier).
  // We keep it as a random blob here to avoid giving proof outside supervised booth.
  const sealed = `sealed_${randHex(18)}`;

  // Short code derived from hC.
  const shortCode = base32Short(hC);

  // QR payload matches UI doc structure: {v,E,s,hC,nonce,sealed}
  const qrObj = { v: 1, E: epoch, s: serial, hC, nonce, sealed };
  const qrPayload = JSON.stringify(qrObj);

  const rec: VoteRecord = {
    voterId: session.voterId,
    constituencyId: session.constituencyId,
    contestId: body.contestId,
    candidateId: body.candidateId,
    txID, epoch, serial, hC, castTime, nonce, sealed,
    shortCode,
    qrPayload
  };

  // last-vote-wins: overwrite voter record; retain serial record for verification history.
  const prev = VOTES_BY_VOTER.get(session.voterId);
  if (prev) {
    // mark old as superseded by leaving it in map; verify endpoint will classify as SUPERSEDED if not latest
  }
  VOTES_BY_VOTER.set(session.voterId, rec);
  VOTES_BY_SERIAL.set(serial, rec);
  SHORT_TO_SERIAL.set(shortCode, serial);

  const resp: CastResp = {
    txID, epoch, serial, hC, castTime, nonce, sealed,
    shortCode, qrPayload,
    smsSent: true,
    printAllowed: session.mode === "kiosk"
  };
  return json(resp);
}

async function handleEnroll(req: Request) {
  const body = (await req.json()) as EnrollReq;
  if (!body.boothId || !body.enrollCode) return json({ error: "missing boothId/enrollCode" }, 400);

  // Demo enrollment rule: accept "BOOTH-17" + "123456"
  if (body.boothId !== "BOOTH-17" || body.enrollCode !== "123456") {
    return forbidden("invalid enrollment");
  }

  const deviceId = `DEV-${randHex(6)}`;
  const deviceToken = `dt_${randHex(18)}`;
  const ttl = 8 * 60 * 60 * 1000; // 8 hours
  const expiresAt = now() + ttl;

  const dev: Device = { deviceId, deviceToken, boothId: body.boothId, expiresAt };
  DEVICES.set(deviceId, dev);

  const resp: EnrollResp = { deviceId, deviceToken, issuedAt: iso(now()), expiresAt: iso(expiresAt) };
  return json(resp);
}

async function handleVerify(req: Request) {
  const deviceToken = getDeviceToken(req);
  if (!deviceToken) return unauthorized("device credential required");

  const body = (await req.json()) as VerifyReq;
  const dev = DEVICES.get(body.deviceId);
  if (!dev) return forbidden("unknown device");
  if (dev.deviceToken !== deviceToken) return forbidden("bad device credential");
  if (dev.expiresAt < now()) return forbidden("device credential expired");

  const key = body.qrOrShortCode?.trim();
  if (!key) return json({ error: "missing qrOrShortCode" }, 400);

  // Interpret input: JSON payload OR short code.
  let qrObj: any | null = null;
  let serial: string | null = null;

  if (key.startsWith("{") && key.endsWith("}")) {
    try {
      qrObj = JSON.parse(key);
      serial = qrObj.s;
    } catch { /* fall through */ }
  }
  if (!serial) {
    serial = SHORT_TO_SERIAL.get(key) || null;
  }
  if (!serial) {
    return json({ status: "INVALID", epoch: 0, serial: "?", hC: "?", castTime: iso(now()), reason: "Unknown receipt", canRevote: true } satisfies VerifyResp);
  }

  const rec = VOTES_BY_SERIAL.get(serial);
  if (!rec) {
    return json({ status: "INVALID", epoch: 0, serial, hC: "?", castTime: iso(now()), reason: "Receipt not found", canRevote: true } satisfies VerifyResp);
  }

  // Determine if superseded: compare with last-vote-wins record.
  const latest = VOTES_BY_VOTER.get(rec.voterId);
  const superseded = latest ? (latest.serial !== rec.serial) : false;

  // Simulate "designated verifier opening sealed": return candidate details only to Client B.
  const ballot = await makeBallot(rec.constituencyId);
  const cand = ballot.candidates.find(c => c.id === latest?.candidateId) || ballot.candidates.find(c => c.id === rec.candidateId);

  if (superseded) {
    return json({
      status: "SUPERSEDED",
      epoch: rec.epoch,
      serial: rec.serial,
      hC: rec.hC,
      castTime: rec.castTime,
      candidate: cand ? { id: cand.id, name: cand.name, party: cand.party } : undefined,
      reason: "A later vote exists for this voter (last-vote-wins).",
      canRevote: true
    } satisfies VerifyResp);
  }

  return json({
    status: "CONFIRMED",
    epoch: rec.epoch,
    serial: rec.serial,
    hC: rec.hC,
    castTime: rec.castTime,
    candidate: cand ? { id: cand.id, name: cand.name, party: cand.party } : undefined,
    canRevote: true
  } satisfies VerifyResp);
}

export function installMockBackend() {
  const realFetch = window.fetch.bind(window);

  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const req = new Request(input, init);
    const url = new URL(req.url, window.location.origin);

    if (!url.pathname.startsWith("/api/")) {
      return realFetch(input, init);
    }

    // Very small latency to simulate network.
    await new Promise(r => setTimeout(r, 250));

    try {
      if (url.pathname === "/api/session/start" && req.method === "POST") return await handleStartSession(req);
      if (url.pathname === "/api/session/end" && req.method === "POST") return await handleEndSession(req);
      if (url.pathname === "/api/ballot" && req.method === "GET") return await handleBallot(req, url);
      if (url.pathname === "/api/liveness" && req.method === "POST") return await handleLiveness(req);
      if (url.pathname === "/api/vote/cast" && req.method === "POST") return await handleCast(req);
      if (url.pathname === "/api/verifier/enroll" && req.method === "POST") return await handleEnroll(req);
      if (url.pathname === "/api/receipt/verify" && req.method === "POST") return await handleVerify(req);

      return notFound();
    } catch (e: any) {
      return json({ error: e?.message || "server error" }, 500);
    }
  };
}
