import { useEffect, useMemo, useState } from "preact/hooks";
import { t, useLang } from "../../i18n";
import { api } from "../../services/api";
import { state } from "../../state";
import { navigate } from "../../router";

/**
 * Client A ballot page:
 * - Loads candidate list from read-only ballot publication API
 * - Presents large, touch-friendly radio options
 * - Provides an explicit review screen before cast
 */
export function A_Ballot() {
  const { lang } = useLang();
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState<"choose" | "review">("choose");
  const [selectedId, setSelectedId] = useState<string>(state.selectedCandidateId || "");

  const session = state.session;
  if (!session) return <div class="card"><h2>Session missing</h2><a href="#/a/start">Go to start</a></div>;
  const liveness = state.liveness;
  if (!liveness?.passed) return <div class="card"><h2>Liveness required</h2><a href="#/a/liveness">Go to liveness</a></div>;

  useEffect(() => {
  (async () => {
    if (state.ballot) return;
    setErr("");
    setBusy(true);
    try {
      // Read-only ballot publication is constituency-scoped and cacheable.
      const b = await api.getBallot(session.constituencyId, session.capabilities.ballotReadToken);
      state.ballot = b;
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  })();
}, [session.sessionId]);

  const ballot = state.ballot;
  const selected = selectedId;
  const selectedCand = useMemo(() => ballot?.candidates.find(c => c.id === selected), [ballot, selected]);

  function proceedReview() {
    if (!selected) { setErr("Select one candidate."); return; }
    setErr("");
    setStep("review");
  }

  async function cast() {
    if (!ballot || !selected) return;
    setErr("");
    setBusy(true);
    try {
      const resp = await api.castVote({
        sessionId: session.sessionId,
        contestId: ballot.contestId,
        candidateId: selected,
        ballotDigest: ballot.digest
      }, session.capabilities.castToken);
      state.cast = resp;
      navigate("/a/receipt");
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1>{t(lang, "ballot_title")}</h1>
      {!ballot && <div class="notice">Loading ballot…</div>}
      {err && <div class="notice"><strong>Error:</strong> {err}</div>}

      {ballot && step === "choose" && (
        <div class="card">
          <h2 class="small">Constituency: <span class="kbd">{ballot.constituencyId}</span></h2>
          <p class="small">Select one candidate. Use touch or mouse. You can review before casting.</p>

          <fieldset style="border:0;padding:0;margin:0">
            <legend class="small" style="color:var(--muted)">Candidates</legend>
            {ballot.candidates.map(c => (
              <label style="display:flex;gap:10px;align-items:center;padding:12px;border:1px solid var(--border);border-radius:12px;margin:8px 0">
                <input
                  type="radio"
                  name="cand"
                  value={c.id}
                  checked={selected === c.id}
                  onChange={() => { setSelectedId(c.id); state.selectedCandidateId = c.id; }}
                  aria-label={`Select ${c.name}`}
                />
                <div>
                  <div style="font-weight:700;font-size:1.15rem">{c.name}</div>
                  <div class="small">{c.party}</div>
                </div>
              </label>
            ))}
          </fieldset>

          <div class="hr"></div>
          <button class="primary" onClick={proceedReview} disabled={busy}>{t(lang, "next")}</button>
          <button class="ghost" onClick={() => navigate("/a/liveness")} style="margin-left:10px">{t(lang, "back")}</button>
        </div>
      )}

      {ballot && step === "review" && selectedCand && (
        <div class="card">
          <h2>{t(lang, "review_title")}</h2>
          <p class="small">
            Review your selection. If correct, cast the vote. After casting, you will receive a QR receipt (no vote choice on Client A).
          </p>
          <div class="notice" aria-live="polite">
            <strong>Selected:</strong> {selectedCand.name} ({selectedCand.party})
          </div>

          <div class="hr"></div>
          <button class="primary" onClick={cast} disabled={busy}>{t(lang, "cast_vote")}</button>
          <button class="ghost" onClick={() => setStep("choose")} style="margin-left:10px">{t(lang, "back")}</button>
        </div>
      )}
    </div>
  );
}
