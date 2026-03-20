/**
 * Landing page that introduces the two client roles in the prototype.
 *
 * The page is not part of the election workflow itself. Its purpose is to make
 * the role split explicit for demonstrations, testing, and reviewer inspection by
 * showing the voter-facing path and the verifier-terminal path side by side.
 */
import { t, useLang } from "../i18n";

export function Home() {
  const { lang } = useLang();
  return (
    <div>
      <h1>{t(lang, "home_title")}</h1>
      <p>{t(lang, "safety_note")}</p>

      <div class="grid2">
        <div class="card">
          <h2>{t(lang, "home_a")}</h2>
          <p class="small">
            Voting client for remote voters or supervised kiosk use. Includes liveness capture, ballot selection, and receipt display.
          </p>
          <a class="badge" href="#/a/start">Open Client A</a>
        </div>

        <div class="card">
          <h2>{t(lang, "home_b")}</h2>
          <p class="small">
            Official verifier app intended for supervised booths only. Requires enrollment (device credential) in this demo.
          </p>
          <a class="badge" href="#/b/verify">Open Client B</a>
          <div class="small" style="margin-top:10px;color:var(--muted)">
            If not enrolled, go to <span class="kbd">/b/enroll</span> first.
          </div>
        </div>
      </div>
    </div>
  );
}
