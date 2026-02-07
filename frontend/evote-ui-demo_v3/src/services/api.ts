/**
 * API wrapper. In production this should call your gateway endpoints.
 * For this demo, `installMockBackend()` intercepts requests to /api/*.
 */

export type Mode = "remote" | "kiosk";

export type SessionStartReq = {
  mode: Mode;
  voterId: string;
  constituencyId: string;
  officerPin?: string; // kiosk only (placeholder for phishing-resistant auth in production)
};

export type SessionStartResp = {
  sessionId: string;
  mode: Mode;
  sessionToken: string;
  expiresAt: string; // ISO timestamp
  constituencyId: string;
  capabilities: {
    ballotReadToken: string;
    castToken: string;
  };
  boothContext?: {
    boothId: string;
    terminal: "A" | "B";
  };
};

export type Candidate = {
  id: string;
  name: string;
  party: string;
  symbolSvgDataUri?: string; // self-hosted or inline; demo uses generated placeholders
};

export type BallotResp = {
  electionId: string;
  constituencyId: string;
  contestId: string;
  digest: string; // integrity hash of ballot definition
  candidates: Candidate[];
};

export type LivenessReq = {
  sessionId: string;
  stillsJpegB64: string[]; // JPEG base64 strings (downscaled frames)
};

export type LivenessResp = {
  passed: boolean;
  score: number;
  dedup: {
    status: "NEW" | "REVOTE";
    prev?: { castTime: string; boothId: string; receiptShortCode: string };
  };
};

export type CastReq = {
  sessionId: string;
  contestId: string;
  candidateId: string;
  ballotDigest: string;
};

export type CastResp = {
  txID: string;
  epoch: number;
  serial: string;
  hC: string; // commitment hash
  castTime: string; // ISO time
  nonce: string;
  sealed: string; // opaque to Client A
  shortCode: string; // human-enterable receipt code
  qrPayload: string; // string encoded into QR (compact JSON here)
  smsSent: boolean;
  printAllowed: boolean;
};

export type VerifyReq = {
  deviceId: string;           // kiosk device identity (Client B)
  deviceToken: string;        // device-bound credential (demo)
  qrOrShortCode: string;      // payload string or short code
};

export type VerifyResp = {
  status: "CONFIRMED" | "PENDING" | "SUPERSEDED" | "INVALID";
  epoch: number;
  serial: string;
  hC: string;
  castTime: string;
  // Candidate details are returned only to Client B in supervised booths.
  candidate?: { id: string; name: string; party: string };
  reason?: string;
  canRevote: boolean;
};

export type EnrollReq = { boothId: string; enrollCode: string };
export type EnrollResp = { deviceId: string; deviceToken: string; issuedAt: string; expiresAt: string };

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${msg || res.statusText}`);
  }
  return await res.json() as T;
}

export const api = {
  startSession: (req: SessionStartReq) => j<SessionStartResp>("/api/session/start", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req)
  }),
  endSession: (sessionId: string, sessionToken: string) => j<{ ok: true }>("/api/session/end", {
    method: "POST",
    headers: { "content-type": "application/json", "authorization": `Bearer ${sessionToken}` },
    body: JSON.stringify({ sessionId })
  }),
  getBallot: (constituencyId: string, ballotReadToken: string) => j<BallotResp>(`/api/ballot?constituencyId=${encodeURIComponent(constituencyId)}`, {
    headers: { "authorization": `Bearer ${ballotReadToken}` }
  }),
  liveness: (req: LivenessReq, sessionToken: string) => j<LivenessResp>("/api/liveness", {
    method: "POST",
    headers: { "content-type": "application/json", "authorization": `Bearer ${sessionToken}` },
    body: JSON.stringify(req)
  }),
  castVote: (req: CastReq, castToken: string) => j<CastResp>("/api/vote/cast", {
    method: "POST",
    headers: { "content-type": "application/json", "authorization": `Bearer ${castToken}` },
    body: JSON.stringify(req)
  }),
  enrollVerifier: (req: EnrollReq) => j<EnrollResp>("/api/verifier/enroll", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req)
  }),
  verifyReceipt: (req: VerifyReq) => j<VerifyResp>("/api/receipt/verify", {
    method: "POST",
    headers: { "content-type": "application/json", "authorization": `Device ${req.deviceToken}` },
    body: JSON.stringify(req)
  })
};
