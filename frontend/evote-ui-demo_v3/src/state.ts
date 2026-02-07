/**
 * Single in-memory store for demo purposes.
 * In production, keep sensitive tokens in memory only (not localStorage) to reduce persistence.
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

/** Clears per-voter state while keeping benign preferences (e.g., language in localStorage) intact. */
export function clearSessionState() {
  delete state.session;
  delete state.ballot;
  delete state.liveness;
  delete state.cast;
  delete state.selectedCandidateId;
}
