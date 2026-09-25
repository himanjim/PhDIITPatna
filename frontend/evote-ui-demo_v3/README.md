# eVote UI Demo (Client A and Client B)

This is a frontend-only demonstration application that implements the voting and
receipt-verification flows described in the user-interface design component of
the thesis. It is built with Vite and Preact and runs entirely in the browser
against a mock backend, so no gateway, peer or network is needed to exercise the
flows.

What it demonstrates:

- a mock backend implemented inside the browser, in `src/services/mockBackend.ts`
- webcam capture with client-side downscaling and JPEG compression
- QR rendering on the voting client and QR scanning on the verifier client, using
  the browser-native `BarcodeDetector` API

## Contents

- `src/`: the Preact application source.
- `docs/`: UI and web API setup notes.
- `perf/`: the inputs and captured outputs of the performance measurements
  reported in the thesis.
- `tools/`: the mock API gateway, the liveness payload generator, the measurement
  driver and the bundle-size budget script.
- `index.html`, `package.json`, `package-lock.json`, `tsconfig.json` and
  `vite.config.ts`: the build and dependency configuration.

## Quick start

```bash
cd frontend/evote-ui-demo_v3
npm install
npm run dev
```

Then open `http://localhost:5173/#/`. The other scripts are `npm run build`,
which writes the production bundle to `dist/`, and `npm run preview`, which
serves that bundle.

The runtime dependencies are `preact` and `qrcode`. Everything else is a build
dependency. `dist/` and `node_modules/` are not committed.

## Demo flows

### Client A, voting

- `#/a/start` starts a session, remote or kiosk.
- `#/a/liveness` captures three frames and runs the dummy liveness and de-duplication step.
- `#/a/ballot` fetches the candidate list, then select, review and cast.
- `#/a/receipt` shows the QR code and short code, with optional print in kiosk mode and an optional hand-off to the verifier.
- `#/a/end` confirms that the session state has been wiped for the next voter.

### Client B, verification

- `#/b/enroll` enrols the device. The demo values are booth `BOOTH-17` and enrolment code `123456`.
- `#/b/verify` scans the QR code or accepts a pasted payload or short code, and displays the confirmed choice, booth only.

## Backend endpoints, mocked

The UI calls the endpoints below and `installMockBackend()` intercepts them:

- `POST /api/session/start`
- `POST /api/session/end`
- `GET  /api/ballot?constituencyId=...`
- `POST /api/liveness`
- `POST /api/vote/cast`
- `POST /api/verifier/enroll`
- `POST /api/receipt/verify`

## Moving to a real backend

- Remove the `installMockBackend()` call from `src/main.tsx` and point `fetch` at
  the real gateway.
- Enforce mutual TLS and device-bound credentials for Client B.
- Enforce a strict content security policy and managed-browser policy in kiosk mode.
- Scope session tokens per operation, separating ballot read from cast write.
- Minimise logging and keep tokens in memory. Do not place session tokens in
  `localStorage`.

## Browser note

QR scanning uses the browser-native `BarcodeDetector` API, which works reliably
in Chromium-based browsers. Where it is unavailable, use the manual paste or
short-code input on Client B.

## Scope

This subtree has a coherent application layout and should not be reorganised
casually. The route filenames encode the flow order, so renaming them requires a
matching change in `src/router.tsx`.
