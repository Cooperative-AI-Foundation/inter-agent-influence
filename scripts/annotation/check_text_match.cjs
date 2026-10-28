// Does jump-to-quote still land once turns are rendered from markdown?
//
// Reads records of {quote, raw, rendered} — rendered being the turn's text
// content as the browser builds it — and compares the matcher the page used to
// run (case-sensitive, on raw text) with the one it runs now (textmatch.js, on
// rendered text). Then simulates searching: markdown pasted from the source,
// and text typed from the screen. Run from the repo root:
//
//   annotation_app/.venv/bin/python scripts/annotation/export_match_records.py bundle/b500 /tmp/records.jsonl
//   node scripts/annotation/check_text_match.cjs /tmp/records.jsonl
const fs = require("fs");
const path = require("path");
const TM = require(path.join(__dirname, "../../annotation_app/app/static/textmatch.js"));

const file = process.argv[2];
const records = fs.readFileSync(file, "utf8").split("\n").filter(Boolean).map(JSON.parse);

// The matcher as it was: whitespace collapse only, case-sensitive, raw text.
const normalise = (s) => s.replace(/\s+/g, " ").trim();
function oldFinds(quote, raw) {
  const flat = normalise(quote);
  const cands = [flat]
    .concat(flat.split(/\.{3}|…/).map(normalise).filter((s) => s.length > 25))
    .concat([flat.split(/\s+/).slice(0, 12).join(" ")])
    .concat([flat.split(/\s+/).slice(0, 6).join(" ")]);
  const hay = normalise(raw);
  return cands.some((c) => c && hay.includes(c));
}
function newFinds(quote, text) {
  const hay = TM.fold(text);
  return TM.candidates(quote, true).some((c) => hay.includes(c));
}

const tally = { total: 0, old_raw: 0, new_rendered: 0, new_raw: 0, lost: [] };
const byScorer = {};
for (const r of records) {
  tally.total++;
  const a = oldFinds(r.quote, r.raw);
  const b = newFinds(r.quote, r.rendered);
  if (a) tally.old_raw++;
  if (b) tally.new_rendered++;
  if (newFinds(r.quote, r.raw)) tally.new_raw++;
  if (a && !b) tally.lost.push(r);
  const s = (byScorer[r.scorer] ||= { n: 0, old: 0, now: 0 });
  s.n++; if (a) s.old++; if (b) s.now++;
}
const pct = (k) => `${tally[k]} (${(100 * tally[k] / tally.total).toFixed(1)}%)`;
console.log(`anchored quotes:                      ${tally.total}`);
console.log(`found before  (old matcher, raw text):  ${pct("old_raw")}`);
console.log(`found now     (new matcher, rendered):  ${pct("new_rendered")}`);
console.log(`  new matcher on raw text, for scale:   ${pct("new_raw")}`);
console.log(`found before but lost now:              ${tally.lost.length}`);
for (const [s, v] of Object.entries(byScorer)) {
  console.log(`  ${s.padEnd(12)} ${v.old}/${v.n} -> ${v.now}/${v.n}`);
}
for (const r of tally.lost) {
  console.log(`\nLOST ${r.row} ${r.scorer}\n  quote:    ${JSON.stringify(r.quote.slice(0, 140))}`);
  const i = TM.fold(r.rendered).indexOf(TM.fold(TM.plainOf(r.quote)).slice(0, 20));
  console.log(`  rendered: ${JSON.stringify(TM.fold(r.rendered).slice(Math.max(0, i - 20), i + 140))}`);
}

// ── typing into the search box ─────────────────────────────────────────
// Someone searching either types what they see or pastes what they copied —
// and a copy can come from the markdown source. For every distinct turn that
// has markdown in it, take eight-word windows of the RAW source that contain
// markdown syntax, and check each is found in the RENDERED text; then the same
// windows as a reader would retype them (markdown stripped), as a control.
const MD = /(\*\*|__|`|^#{1,6} |^\s*[-*+] |^\s*\d+\. |\|)/m;
const seen = new Set();
let asPasted = 0, asTyped = 0, windows = 0;
const misses = [];
for (const r of records) {
  const key = r.row + "|" + r.raw.length + "|" + r.raw.slice(0, 40);
  if (seen.has(key) || !MD.test(r.raw)) continue;
  seen.add(key);
  const hay = TM.fold(r.rendered);
  const words = r.raw.split(/(?<=\s)/);        // keep the whitespace, newlines too
  for (let i = 0; i + 8 <= words.length; i += 5) {
    const win = words.slice(i, i + 8).join("");
    if (!MD.test(win)) continue;
    windows++;
    const pasted = TM.candidates(win, false).some((c) => hay.includes(c));

    if (pasted) asPasted++; else if (misses.length < 6) misses.push(win);
  }
}
const p = (n) => `${n}/${windows} (${(100 * n / windows).toFixed(1)}%)`;
console.log(`\nsearch windows containing markdown syntax: ${windows}`);
console.log(`  pasted from the source, found:   ${p(asPasted)}`);

// ...and what someone types from the screen: eight-word windows of the
// rendered text itself. This should be every one — it checks that the
// search's folding agrees with the page's text index.
let screen = 0, screenHit = 0;
for (const r of records) {
  const words = r.rendered.split(/(?<=\s)/);
  for (let i = 0; i + 8 <= words.length; i += 23) {
    const win = words.slice(i, i + 8).join("");
    if (TM.fold(win).length < 2) continue;
    screen++;
    if (TM.candidates(win, false).some((c) => TM.fold(r.rendered).includes(c))) screenHit++;
  }
}
console.log(`  typed from the screen, found:    ${screenHit}/${screen} (${(100 * screenHit / screen).toFixed(1)}%)`);
for (const m of misses) console.log(`  miss: ${JSON.stringify(m)}`);
