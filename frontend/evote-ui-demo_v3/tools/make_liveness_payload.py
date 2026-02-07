import argparse, base64, json
from io import BytesIO
from PIL import Image

def center_square(img):
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def encode_jpeg_b64(path, size=320, quality=60):
    img = Image.open(path).convert("RGB")
    img = center_square(img).resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    jpeg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return jpeg_bytes, b64

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
