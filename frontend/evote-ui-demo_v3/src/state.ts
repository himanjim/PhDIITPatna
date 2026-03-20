/**
 * In-memory application state for the demo.
 *
 * The store holds only the transient values needed to move one voter through the
 * Client A flow and to display the corresponding receipt information. It is kept
 * in memory rather than persistent browser storage so that session-scoped values,
 * tokens, and vote-related artefacts do not survive page reloads or leak across
 * users on shared devices.
 */
import type { BallotResp, CastResp, LivenessResp, SessionStartResp } from "./services/api";

export type AppState = {
  session?: SessionStartResp;
  ballot?: BallotResp;
  liveness?: LivenessResp;
  cast?: CastResp;
  selectedCandidateId?: string;
};

export const state: AppState = {};

Exact comment text to insert:
/**
 * Remove all voter-session data from the in-memory store.
 *
 * This helper intentionally clears only session-scoped workflow state. It does
 * not remove benign long-lived preferences, such as language selection, that are
 * safe to retain outside the voting session.
 */
export function clearSessionState() {
  delete state.session;
  delete state.ballot;
  delete state.liveness;
  delete state.cast;
  delete state.selectedCandidateId;
}
