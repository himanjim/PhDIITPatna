# eVote UI Demo (Client A + Client B)

This is a **frontend-only** demo that implements the flows described in `User-interface design_9.docx` using:
- **Vite + Preact** (thin framework)
- A **mock backend** implemented inside the browser (`src/services/mockBackend.ts`)
- Working webcam capture + downscale + JPEG compression
- QR rendering (Client A) and QR scanning via `BarcodeDetector` (Client B, Chromium)

## Quick start
1. Unzip the project.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run dev server:
   ```bash
   npm run dev
   ```
4. Open `http://localhost:5173/#/`

## Demo flows
### Client A (Voting)
- `#/a/start` → start session (remote or kiosk)
- `#/a/liveness` → capture 3 frames; dummy liveness + dedup
- `#/a/ballot` → fetch ballot (candidate list), select, review, cast
- `#/a/receipt` → show QR + short code, optional print (kiosk), optional proceed to verifier
- `#/a/end` → confirms state wipe for next voter

### Client B (Verifier)
- `#/b/enroll` → enroll device (demo: boothId `BOOTH-17`, enrollCode `123456`)
- `#/b/verify` → scan QR or paste payload/short code, view confirmed choice (booth-only)

## Backend endpoints (mocked)
The UI calls these endpoints; the mock intercepts them in `installMockBackend()`:
- `POST /api/session/start`
- `POST /api/session/end`
- `GET  /api/ballot?constituencyId=...`
- `POST /api/liveness`
- `POST /api/vote/cast`
- `POST /api/verifier/enroll`
- `POST /api/receipt/verify`

## Production notes (what to swap out)
- Delete `installMockBackend()` from `src/main.tsx` and point `fetch` to your real gateway.
- Enforce:
  - mTLS / device-bound credentials for Client B
  - strict CSP, kiosk managed-browser policies
  - per-session token scopes: ballot-read vs cast-write
  - logging minimization, in-memory tokens (no localStorage for session tokens)


## Browser note
- QR scanning uses the browser-native `BarcodeDetector` API, which works reliably in Chromium-based browsers.
- If QR scanning is unavailable, use the manual paste/short-code input in Client B.
