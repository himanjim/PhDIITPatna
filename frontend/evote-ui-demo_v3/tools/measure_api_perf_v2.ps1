param(
  [string]$BaseUrl = "http://127.0.0.1:8080",
  [int]$Runs = 30,
  [string]$Mode = "remote"
)

function Percentile([double[]]$xs, [double]$p) {
  if ($xs.Count -eq 0) { return [double]::NaN }
  $s = $xs | Sort-Object
  $n = $s.Count
  $idx = ($p/100.0) * ($n - 1)
  $lo = [math]::Floor($idx); $hi = [math]::Ceiling($idx)
  if ($lo -eq $hi) { return [double]$s[$lo] }
  $w = $idx - $lo
  return (1-$w)*[double]$s[$lo] + $w*[double]$s[$hi]
}

# Inputs
$constituencies = Get-Content "perf\constituencies.txt" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
if ($constituencies.Count -eq 0) { throw "perf\constituencies.txt is empty" }
foreach ($p in @("perf\stills\1.jpg","perf\stills\2.jpg","perf\stills\3.jpg")) { if (!(Test-Path $p)) { throw "Missing $p" } }

# Ensure python liveness payload builder exists
if (!(Test-Path "tools\make_liveness_payload.py")) { throw "Missing tools\make_liveness_payload.py (your earlier script creates it). Recreate it or tell me and I’ll paste it again." }

"run,cid,ballot_encoding,ballot_download_bytes,frame1_jpeg_bytes,frame2_jpeg_bytes,frame3_jpeg_bytes,liveness_json_bytes,liveness_upload_bytes,liveness_time_s,cast_time_s" | Set-Content -Encoding UTF8 "perf\api_measurements.csv"

$ballotKB = @()
$frameBytes = @()
$livenessUploadKB = @()
$castTimes = @()

for ($i=1; $i -le $Runs; $i++) {
  $cid = $constituencies[($i-1) % $constituencies.Count]
  $voterId = "VOTER-" + (Get-Random -Minimum 100000 -Maximum 999999) + "-$i"

  # ---- 1) START SESSION (file-based)
  $startFile = "perf\start_$i.json"
  @"
{"mode":"$Mode","voterId":"$voterId","constituencyId":"$cid"}
"@ | Set-Content -Encoding UTF8 $startFile

  $startJson = curl.exe -sS --fail-with-body -X POST "$BaseUrl/api/session/start" `
    -H "Content-Type: application/json" `
    --data-binary "@$startFile"

  $sess = $startJson | ConvertFrom-Json
  if (-not $sess.sessionId) { throw "Session start failed. Response: $startJson" }

  $sessionId = $sess.sessionId
  $sessionToken = $sess.sessionToken
  $ballotReadToken = $sess.capabilities.ballotReadToken
  $castToken = $sess.capabilities.castToken

  # ---- 2) BALLOT (measure compressed download bytes)
  $hdr = New-TemporaryFile
  $ballotPath = "perf\ballot_$i.json"

  $ballotBytes = curl.exe -sS --fail-with-body --compressed -D $hdr -o $ballotPath -w "%{size_download}" `
    -H "Authorization: Bearer $ballotReadToken" `
    "$BaseUrl/api/ballot?constituencyId=$cid"

  $encLine = (Get-Content $hdr | Select-String -Pattern '^Content-Encoding:' | Select-Object -First 1).Line
  $encoding = if ($encLine) { $encLine.Split(":",2)[1].Trim() } else { "identity" }
  Remove-Item $hdr -Force

  $ballotRaw = Get-Content $ballotPath -Raw
  try { $ballot = $ballotRaw | ConvertFrom-Json } catch { throw "Ballot response not JSON. File: $ballotPath. Content (first 200 chars): $($ballotRaw.Substring(0,[Math]::Min(200,$ballotRaw.Length)))" }

  $contestId = $ballot.contestId
  $digest = $ballot.digest
  if (-not $ballot.candidates -or $ballot.candidates.Count -lt 1) { throw "No candidates in ballot: $ballotPath" }
  $candidateId = $ballot.candidates[0].id

  # ---- 3) BUILD LIVENESS PAYLOAD (file)
  $livenessPayloadPath = "perf\liveness_$i.json"
  $pyOut = python tools\make_liveness_payload.py `
    --sessionId "$sessionId" `
    --out "$livenessPayloadPath" `
    --size 320 `
    --quality 0.6 `
    --stills "perf\stills\1.jpg" "perf\stills\2.jpg" "perf\stills\3.jpg"

  if ($LASTEXITCODE -ne 0) { throw "Python liveness payload builder failed for run $i." }
  $ls = $pyOut | ConvertFrom-Json

  $f1 = [int]$ls.jpeg_bytes_per_still[0]
  $f2 = [int]$ls.jpeg_bytes_per_still[1]
  $f3 = [int]$ls.jpeg_bytes_per_still[2]
  $livenessJsonBytes = [int]$ls.json_body_bytes

  # ---- 4) POST LIVENESS (measure upload bytes + time)
  $tmpHdr2 = New-TemporaryFile
  $livenessMeta = curl.exe -sS --fail-with-body --compressed -D $tmpHdr2 -o NUL `
    -w "%{size_upload} %{time_total}" `
    -H "Content-Type: application/json" `
    -H "Authorization: Bearer $sessionToken" `
    --data-binary "@$livenessPayloadPath" `
    "$BaseUrl/api/liveness"
  Remove-Item $tmpHdr2 -Force

  if (-not $livenessMeta) { throw "Liveness curl returned empty meta for run $i." }
  $parts = $livenessMeta.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
  if ($parts.Count -lt 2) { throw "Unexpected liveness meta '$livenessMeta' for run $i." }
  $livenessUploadBytes = [int]$parts[0]
  $livenessTime = [double]$parts[1]

  # ---- 5) CAST (file-based) + time_total
  $castFile = "perf\cast_$i.json"
  @"
{"sessionId":"$sessionId","contestId":"$contestId","candidateId":"$candidateId","ballotDigest":"$digest"}
"@ | Set-Content -Encoding UTF8 $castFile

  $castTime = curl.exe -sS --fail-with-body -o NUL -w "%{time_total}" `
    -X POST "$BaseUrl/api/vote/cast" `
    -H "Content-Type: application/json" `
    -H "Authorization: Bearer $castToken" `
    --data-binary "@$castFile"
  $castTime = [double]$castTime

  # ---- Record row
  "$i,$cid,$encoding,$ballotBytes,$f1,$f2,$f3,$livenessJsonBytes,$livenessUploadBytes,$livenessTime,$castTime" | Add-Content -Encoding UTF8 "perf\api_measurements.csv"

  # ---- Collect stats
  $ballotKB += ([double]$ballotBytes/1024.0)
  $frameBytes += $f1; $frameBytes += $f2; $frameBytes += $f3
  $livenessUploadKB += ([double]$livenessUploadBytes/1024.0)
  $castTimes += $castTime
}

# ---- Summary
$summary = @()
$summary += "Ballot payload (compressed download):"
$summary += ("  p50 = {0:n1} KB, p95 = {1:n1} KB, max = {2:n1} KB" -f (Percentile $ballotKB 50),(Percentile $ballotKB 95),(($ballotKB | Measure-Object -Maximum).Maximum))

$summary += ""
$summary += "Liveness still JPEG bytes (per still; 320x320, JPEG q=0.6):"
$framesKB = $frameBytes | ForEach-Object { [double]$_/1024.0 }
$summary += ("  p50 = {0:n1} KB, p95 = {1:n1} KB, max = {2:n1} KB" -f (Percentile $framesKB 50),(Percentile $framesKB 95),(($framesKB | Measure-Object -Maximum).Maximum))

$summary += ""
$summary += "Liveness upload (HTTP upload bytes; includes JSON/base64 overhead):"
$summary += ("  p50 = {0:n1} KB, p95 = {1:n1} KB, max = {2:n1} KB" -f (Percentile $livenessUploadKB 50),(Percentile $livenessUploadKB 95),(($livenessUploadKB | Measure-Object -Maximum).Maximum))

$summary += ""
$summary += "Cast confirmation time (API /vote/cast time_total):"
$summary += ("  p50 = {0:n2} s, p95 = {1:n2} s, max = {2:n2} s" -f (Percentile $castTimes 50),(Percentile $castTimes 95),(($castTimes | Measure-Object -Maximum).Maximum))

$summary -join "`r`n" | Set-Content -Encoding UTF8 "perf\summary.txt"
Get-Content "perf\summary.txt"
