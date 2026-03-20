"""
Build a compact liveness-request payload from a small set of still images.

The script centre-crops each source image, resizes it to a fixed square, JPEG-
encodes it at the requested quality, base64-encodes the result, and writes the
final JSON body expected by the mock liveness endpoint. It also emits size
statistics to stdout so that upload overhead can be measured separately from the
application flow.
"""

import argparse, base64, json
from io import BytesIO
from PIL import Image

"""Return the largest centred square crop from the supplied image."""
def center_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

"""
Load one image, apply the standard centre crop and resize policy, encode it as a
JPEG, and return both the binary JPEG bytes and the ASCII base64 text.
"""
def encode_jpeg_b64(path, size=320, quality=60):
    img = Image.open(path).convert("RGB")
    img = center_square(img).resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    jpeg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return jpeg_bytes, b64

"""
Parse command-line arguments, build the JSON liveness body, write it to disk, and
print machine-readable size statistics for later performance analysis.
"""
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessionId", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--quality", type=float, default=0.6)
    ap.add_argument("--stills", nargs="+", required=True)
    args = ap.parse_args()

    quality_int = max(1, min(95, int(round(args.quality * 100))))
    jpeg_sizes = []
    b64_list = []
    for p in args.stills:
        jpeg_bytes, b64 = encode_jpeg_b64(p, size=args.size, quality=quality_int)
        jpeg_sizes.append(len(jpeg_bytes))
        b64_list.append(b64)

    payload = {"sessionId": args.sessionId, "stillsJpegB64": b64_list}
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with open(args.out, "wb") as f:
        f.write(body)

    # Machine-readable stats to stdout (single line JSON)
    print(json.dumps({
        "jpeg_bytes_per_still": jpeg_sizes,
        "json_body_bytes": len(body),
        "b64_chars_per_still": [len(x) for x in b64_list]
    }))

if __name__ == "__main__":
    main()
