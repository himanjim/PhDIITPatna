import { useMemo } from "preact/hooks";
import { t, useLang } from "../i18n";
/**
 * Shared page header for both client roles.
 *
 * The header presents the application identity and exposes the language selector
 * that controls the interface dictionary for the current browser context. Because
 * language preference is benign and non-vote-specific, it can safely remain
 * available across both Client A and Client B views.
 */
export function Header() {
  const { lang, setLang } = useLang();
  // Additional languages can be added here as dictionary coverage expands.
  const langs = useMemo(() => ([
    { code: "en", label: "English" },
    { code: "hi", label: "हिन्दी" },
    { code: "bn", label: "বাংলা" },
    { code: "ta", label: "தமிழ்" },
    { code: "te", label: "తెలుగు" },
    { code: "mr", label: "मराठी" },
    // NOTE: Add remaining Eighth Schedule languages here as needed.
  ]), []);

  return (
    <div class="card" style="margin:0;border-radius:0;border-left:0;border-right:0">
      <div class="row" style="justify-content:space-between;align-items:center">
        <div>
          <div class="badge">eVote UI Demo</div>
          <div class="small" style="color:var(--muted)">
            {t(lang, "subtitle")}
          </div>
        </div>
        <div style="min-width:220px">
          <label class="small" for="langSel">{t(lang, "language")}</label>
          <select id="langSel" value={lang} onChange={(e) => setLang((e.target as HTMLSelectElement).value as any)}>
            {langs.map(l => <option value={l.code}>{l.label}</option>)}
          </select>
        </div>
      </div>
    </div>
  );
}
