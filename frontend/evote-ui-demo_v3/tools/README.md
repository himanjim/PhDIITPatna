# Frontend tools

This directory contains helper scripts used to exercise or measure the frontend
demo outside the application bundle itself.

## Contents

- `measure_api_perf_v2.ps1` is the canonical end-to-end measurement driver. It
  writes the start-session, liveness and cast payloads to files before submitting
  them, so that the measured sizes correspond to explicit artefacts on disk. It
  records per-run transfer sizes and timings to `perf/api_measurements.csv` and
  writes the distribution summary to `perf/summary.txt`. This is the script the
  user-interface component of the thesis cites, and the committed measurement
  artefacts come from it.
- `measure_api_perf.ps1` is the earlier version, retained for reference. It
  measures the same endpoints but submits the payloads in memory rather than from
  files, and it writes to the same two output paths. Running it will therefore
  overwrite the artefacts produced by the v2 script. Use it only when the
  in-memory measurement path is the point of the experiment.
- `mock_api_server.cjs` is the mock API gateway used for controlled measurement.
- `make_liveness_payload.py` generates deterministic liveness payloads that
  reproduce the client's downscale and JPEG encoding policy.
- `perf_budget.cjs` checks the built bundle against the size budget.

## Running a measurement

Start the mock gateway, then run the driver from the application root so that the
relative `perf\` paths resolve:

```powershell
node tools\mock_api_server.cjs
.\tools\measure_api_perf_v2.ps1 -BaseUrl http://127.0.0.1:8080 -Runs 30 -Mode remote
```

Both scripts read the constituency list from `perf/constituencies.txt` and fail
if it is empty.

## Notes

- Keep `measure_api_perf_v2.ps1` as the cited driver. If the earlier script is
  ever run, restore the committed artefacts afterwards or note in the commit
  message which script produced them, because the two write to the same files.
