const http = require("http");
const url = require("url");
const zlib = require("zlib");

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 8080;

function json(res, obj, req) {
  const body = Buffer.from(JSON.stringify(obj));
  const ae = (req.headers["accept-encoding"] || "");
  if (/\bbr\b/.test(ae)) {
    const br = zlib.brotliCompressSync(body);
    res.writeHead(200, { "Content-Type":"application/json", "Content-Encoding":"br", "Vary":"Accept-Encoding" });
    return res.end(br);
  }
  if (/\bgzip\b/.test(ae)) {
    const gz = zlib.gzipSync(body, { level: 9 });
    res.writeHead(200, { "Content-Type":"application/json", "Content-Encoding":"gzip", "Vary":"Accept-Encoding" });
    return res.end(gz);
  }
  res.writeHead(200, { "Content-Type":"application/json" });
  return res.end(body);
}

function bad(res, code, msg) {
  res.writeHead(code, { "Content-Type":"application/json" });
  res.end(JSON.stringify({ error: msg }));
}

const sessions = new Map(); // sessionId -> { voterId, constituencyId, mode }
const votesByVoter = new Map(); // voterId -> { candidateId, ts }

function rand(prefix="") {
  return prefix + Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", c => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function cleanJsonText(raw) {
  return (raw || "").toString("utf-8").replace(/^\uFEFF/, "").trim();
}

const server = http.createServer(async (req, res) => {
  const u = url.parse(req.url, true);
  const p = u.pathname || "/";

  if (req.method === "GET" && p === "/health") return json(res, { ok: true }, req);

  // START SESSION
  if (req.method === "POST" && p === "/api/session/start") {
    const raw = (await readBody(req)).toString("utf-8") || "{}";
    const cleaned = raw.replace(/^\uFEFF/, "").trim(); // <-- BOM + whitespace safe
    let body = {};
    try {
        body = JSON.parse(cleaned);
    } catch (e) {
    console.error("Bad JSON on /api/session/start. Raw body was:\n", raw);
    res.writeHead(400, { "Content-Type": "application/json" });
    return res.end(JSON.stringify({
      error: "invalid JSON",
      hint: "Send JSON (BOM-free) with quoted keys/values",
      raw
    }));
    }
    const mode = body.mode || "remote";
    const voterId = body.voterId || rand("VOTER-");
    const constituencyId = body.constituencyId || "PATNA-01";
    const sessionId = rand("sess_");
    const sessionToken = rand("st_");
    const ballotReadToken = rand("br_");
    const castToken = rand("ct_");
    sessions.set(sessionId, { voterId, constituencyId, mode, sessionToken, ballotReadToken, castToken });

    return json(res, {
      sessionId,
      sessionToken,
      capabilities: { ballotReadToken, castToken }
    }, req);
  }

  // BALLOT
  if (req.method === "GET" && p === "/api/ballot") {
    const constituencyId = u.query.constituencyId || "PATNA-01";
    // Minimal realistic payload: text + IDs; pictograms are assumed local.
    const candidates = [
      { id: "cand_01", name: "Candidate A", party: "Party X", symbolKey: "SYMBOL_X" },
      { id: "cand_02", name: "Candidate B", party: "Party Y", symbolKey: "SYMBOL_Y" },
      { id: "cand_03", name: "Candidate C", party: "Party Z", symbolKey: "SYMBOL_Z" }
    ];
    return json(res, {
      contestId: "LS2029-" + constituencyId,
      constituencyId,
      digest: rand("dig_"),
      candidates
    }, req);
  }

  // LIVENESS
  if (req.method === "POST" && p === "/api/liveness") {
    let payload;
    try {
      const raw = await readBody(req);
      payload = JSON.parse(cleanJsonText(raw));
    } catch {
      return bad(res, 400, "invalid JSON");
    }
    const sessionId = payload.sessionId;
    const s = sessions.get(sessionId);
    if (!s) return bad(res, 401, "unknown session");
    const stills = payload.stillsJpegB64 || [];
    const ok = Array.isArray(stills) && stills.length >= 2 && stills[0].length > 1000;

    const already = votesByVoter.has(s.voterId);
    const dedupStatus = already ? "REVOTE" : "OK";

    return json(res, {
      ok,
      livenessScore: ok ? 0.92 : 0.12,
      dedup: { status: dedupStatus, matchConfidence: already ? 0.98 : 0.0 }
    }, req);
  }

  // CAST VOTE
  if (req.method === "POST" && p === "/api/vote/cast") {
    let payload;
    try {
      const raw = await readBody(req);
      payload = JSON.parse(cleanJsonText(raw));
    } catch {
      return bad(res, 400, "invalid JSON");
    }
    const sessionId = payload.sessionId;
    const s = sessions.get(sessionId);
    if (!s) return bad(res, 401, "unknown session");

    const candidateId = payload.candidateId || "cand_01";
    votesByVoter.set(s.voterId, { candidateId, ts: Date.now() });

    // Receipt shape similar to your UI demo
    const receipt = {
      serial: rand("SER_"),
      epoch: Math.floor(Date.now() / 1000),
      shortCode: (Math.floor(100000 + Math.random()*900000)).toString(),
      qrPayload: {
        v: 1,
        contestId: payload.contestId || ("LS2029-" + s.constituencyId),
        serial: rand("SER_"),
        hC: rand("hC_"),
        txHint: rand("tx_"),
        nonce: rand("n_")
      }
    };

    return json(res, receipt, req);
  }

  // Default
  bad(res, 404, `no route: ${req.method} ${p}`);
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Mock API listening on http://127.0.0.1:${PORT}`);
});
