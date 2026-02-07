import { useEffect, useState } from "preact/hooks";

/**
 * Minimal i18n.
 * - Document requires Eighth Schedule coverage; this demo includes 6 languages.
 * - Any missing string falls back to English.
 */
export type Lang = "en" | "hi" | "bn" | "ta" | "te" | "mr";

type Dict = Record<string, string>;

const EN: Dict = {
  subtitle: "Client A (Vote) + Client B (Verifier) — with mock backend",
  language: "Language",
  home_title: "Select client",
  home_a: "Client A (Voting)",
  home_b: "Client B (Verifier)",
  safety_note: "Demo: backend calls are mocked in-browser.",
  next: "Next",
  back: "Back",
  cancel: "Cancel",
  start_session: "Start session",
  mode: "Mode",
  mode_remote: "Remote voter (personal device)",
  mode_kiosk: "Supervised booth (kiosk)",
  officer_pin: "Officer PIN (kiosk only)",
  voter_id: "Voter ID",
  constituency: "Constituency",
  consent_camera: "Enable camera",
  liveness_title: "Liveness capture (few frames)",
  liveness_help: "We will capture 3 still frames (downscaled) and run a dummy liveness check.",
  capture: "Capture",
  capturing: "Capturing…",
  liveness_ok: "Liveness passed",
  liveness_fail: "Liveness failed",
  revote_flag: "Re-vote case detected",
  dispute: "Dispute",
  officer_confirm: "Officer confirm",
  ballot_title: "Ballot",
  review_title: "Review",
  cast_vote: "Cast vote",
  receipt_title: "Receipt",
  show_qr: "Show QR",
  print_receipt: "Print receipt",
  proceed_to_verify: "Proceed to verification (optional)",
  skip_verify: "Skip verification",
  end_title: "Session ended",
  end_note: "All session state has been cleared.",
  enroll_title: "Client B enrollment (booth only)",
  enroll_code: "Enrollment code",
  enroll: "Enroll device",
  verify_title: "Verify receipt (Client B)",
  scan_qr: "Scan QR",
  paste_qr: "Or paste receipt payload",
  verify: "Verify",
  status: "Status",
  confirmed: "CONFIRMED",
  pending: "PENDING",
  superseded: "SUPERSEDED",
  invalid: "INVALID",
  request_revote: "Request re-vote (return to officer)",
  not_provisioned: "This verifier is not provisioned on this device.",
  go_enroll: "Go to enrollment"
};

const HI: Dict = {
  ...EN,
  subtitle: "क्लाइंट A (मतदान) + क्लाइंट B (सत्यापन) — डेमो",
  language: "भाषा",
  home_title: "क्लाइंट चुनें",
  home_a: "क्लाइंट A (मतदान)",
  home_b: "क्लाइंट B (सत्यापन)",
  start_session: "सत्र शुरू करें",
  mode_remote: "रिमोट मतदाता (व्यक्तिगत डिवाइस)",
  mode_kiosk: "सुपरवाइज़्ड बूथ (कियोस्क)",
  voter_id: "मतदाता आईडी",
  liveness_title: "लाइवनेस कैप्चर",
  ballot_title: "मतपत्र",
  receipt_title: "रसीद",
  verify_title: "रसीद सत्यापित करें"
};
const BN: Dict = { ...EN, language: "ভাষা", home_title: "ক্লায়েন্ট নির্বাচন করুন" };
const TA: Dict = { ...EN, language: "மொழி", home_title: "கிளையண்டை தேர்ந்தெடுக்கவும்" };
const TE: Dict = { ...EN, language: "భాష", home_title: "క్లయింట్ ఎంచుకోండి" };
const MR: Dict = { ...EN, language: "भाषा", home_title: "क्लायंट निवडा" };

const DICTS: Record<Lang, Dict> = { en: EN, hi: HI, bn: BN, ta: TA, te: TE, mr: MR };

export function t(lang: Lang, key: string): string {
  return (DICTS[lang] && DICTS[lang][key]) || EN[key] || key;
}

const LANG_KEY = "evote.lang";

/** Hook for global language selection (stored in localStorage). */
export function useLang() {
  const [lang, setLangState] = useState<Lang>(() => (localStorage.getItem(LANG_KEY) as Lang) || "en");
  useEffect(() => { localStorage.setItem(LANG_KEY, lang); }, [lang]);
  return { lang, setLang: setLangState };
}
