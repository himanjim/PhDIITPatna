const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const { brotliCompressSync, gzipSync, constants } = zlib;

const dist = path.join(process.cwd(), "dist");
const assetsDir = path.join(dist, "assets");
if (!fs.existsSync(dist)) { console.error("dist/ not found. Run: npm run build"); process.exit(1); }

function walk(dir) {
  const out = [];
  for (const name of fs.readdirSync(dir)) {
    const p = path.join(dir, name);
    const st = fs.statSync(p);
    if (st.isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out;
}

function fmt(n) { return (n/1024).toFixed(1) + " KB"; }

const files = [];
const roots = [];
if (fs.existsSync(path.join(dist, "index.html"))) roots.push(path.join(dist, "index.html"));
if (fs.existsSync(assetsDir)) roots.push(...walk(assetsDir));

for (const f of roots) {
  const buf = fs.readFileSync(f);
  const raw = buf.length;
  const gz = gzipSync(buf, { level: 9 }).length;
  const br = brotliCompressSync(buf, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }).length;
  const rel = path.relative(process.cwd(), f).replace(/\\/g,"/");
  const ext = path.extname(f).toLowerCase();
  files.push({ file: rel, ext, raw, gzip: gz, brotli: br });
}

const sum = (pred) => files.filter(pred).reduce((a,x)=>a+x.raw,0);
const sumG = (pred) => files.filter(pred).reduce((a,x)=>a+x.gzip,0);
const sumB = (pred) => files.filter(pred).reduce((a,x)=>a+x.brotli,0);

const jsPred = x => x.ext === ".js";
const cssPred = x => x.ext === ".css";
const htmlPred = x => x.ext === ".html";
const otherPred = x => ![".js",".css",".html"].includes(x.ext);

const totals = {
  html: { raw: sum(htmlPred), gzip: sumG(htmlPred), brotli: sumB(htmlPred) },
  js:   { raw: sum(jsPred),   gzip: sumG(jsPred),   brotli: sumB(jsPred) },
  css:  { raw: sum(cssPred),  gzip: sumG(cssPred),  brotli: sumB(cssPred) },
  other:{ raw: sum(otherPred),gzip: sumG(otherPred),brotli: sumB(otherPred) },
};
totals.all = {
  raw: totals.html.raw + totals.js.raw + totals.css.raw + totals.other.raw,
  gzip: totals.html.gzip + totals.js.gzip + totals.css.gzip + totals.other.gzip,
  brotli: totals.html.brotli + totals.js.brotli + totals.css.brotli + totals.other.brotli,
};

const perfDir = path.join(process.cwd(), "perf");
fs.writeFileSync(path.join(perfDir, "bundle_sizes.json"), JSON.stringify({ files, totals }, null, 2));

let md = "";
md += "| Category | Raw | Gzip | Brotli |\n";
md += "|---|---:|---:|---:|\n";
for (const k of ["html","js","css","other","all"]) {
  const t = totals[k];
  md += `| ${k.toUpperCase()} | ${fmt(t.raw)} | ${fmt(t.gzip)} | ${fmt(t.brotli)} |\n`;
}
md += "\n\nTop assets (by Brotli):\n\n";
md += "| File | Brotli |\n|---|---:|\n";
files.slice().sort((a,b)=>b.brotli-a.brotli).slice(0,10).forEach(x=>{
  md += `| ${x.file} | ${fmt(x.brotli)} |\n`;
});
fs.writeFileSync(path.join(perfDir, "bundle_sizes.md"), md);

console.log("\nWrote:");
console.log("  perf/bundle_sizes.json");
console.log("  perf/bundle_sizes.md\n");
console.log(md);
