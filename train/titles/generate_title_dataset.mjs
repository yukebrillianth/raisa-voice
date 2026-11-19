#!/usr/bin/env node
// Grows the title lexicon with an LLM, then lets other LLMs check every row blind.
//
// Why two steps: a generator at high temperature finds many titles but also invents some.
// Each judge sees only the abbreviation (never the generator's answer) and gives its own
// expansion at temperature 0. A row is accepted only when every judge says it is real and
// all expansions agree. Everything else goes to review.jsonl for a human.
//
//   node generate_title_dataset.mjs generate --config llm.json --per-category 200
//   node generate_title_dataset.mjs verify   --config llm.json
//   node generate_title_dataset.mjs export
//   python build_title_lexicon.py --accepted out/accepted.jsonl --report out/build
//
// llm.json (keys may be "$ENV_NAME"):
//   {
//     "generator": {"baseUrl": "https://generativelanguage.googleapis.com/v1beta/openai", "apiKey": "$GEMINI_API_KEY", "model": "gemini-4-carbon"},
//     "judges": [
//       {"name": "carbon", "baseUrl": "...", "apiKey": "$GEMINI_API_KEY", "model": "gemini-4-carbon"},
//       {"name": "gemma",  "baseUrl": "http://localhost:8080/v1", "apiKey": "x", "model": "gemma-4-12b"}
//     ]
//   }
// Use judges from different model families: two copies of one model share its mistakes.
//
// --grounding lampiran.txt (repeatable) puts official text in the generator prompt, e.g. the
// appendix of Permendikbudristek No. 6 Tahun 2022 or a rector's decree listing degrees per study
// programme. With --strict the generator may only return titles found in that text.
//
// Node 18+, no dependencies. Every step appends to its file and skips work already done,
// so a crash or Ctrl-C loses at most the batch in flight.

import fs from "node:fs";
import path from "node:path";

const CATEGORIES = [
  {
    id: "field",
    type: "field",
    ask: "kode bidang ilmu yang dipakai setelah S., S.Tr., M., M.Tr., A.Md. pada gelar Indonesia",
    example: '{"code": "Kom", "id": "Komputer", "en": "Computer Science", "example": "S.Kom."}',
  },
  {
    id: "spesialis",
    type: "spesialis",
    ask: "kode spesialis dokter dan dokter gigi setelah Sp. (Sp.PD, Sp.A, Sp.Ort ...)",
    example: '{"code": "PD", "id": "Penyakit Dalam", "en": "Internal Medicine", "example": "Sp.PD"}',
  },
  {
    id: "konsultan",
    type: "konsultan",
    ask: "kode konsultan subspesialis yang ditulis setelah -K pada Sp.PD (Sp.PD-KGEH, Sp.PD-KHOM ...)",
    example: '{"code": "GEH", "id": "Gastroenterohepatologi", "en": "Gastroenterohepatology", "example": "Sp.PD-KGEH"}',
  },
  {
    id: "post_foreign",
    type: "post",
    ask: "gelar luar negeri yang sering dipakai dosen Indonesia (AS, Inggris, Australia, Jerman, Belanda, Prancis, Jepang, Malaysia)",
    example: '{"surface": "M.Sc.Eng.", "id": "Master of Science in Engineering", "en": "Master of Science in Engineering", "letters": "en"}',
  },
  {
    id: "post_profession",
    type: "post",
    ask: "gelar profesi, sertifikasi, dan fellowship yang ditulis di belakang nama di Indonesia (IPU, ASEAN Eng., CPA, FINASIM ...)",
    example: '{"surface": "IPU", "id": "Insinyur Profesional Utama", "en": "Insinyur Profesional Utama", "letters": "id"}',
  },
  {
    id: "pre",
    type: "pre",
    ask: "gelar dan sebutan yang ditulis di depan nama di Indonesia: akademik, profesi, keagamaan, kehormatan, adat",
    example: '{"surface": "Prof.", "id": "Profesor", "en": "Professor"}',
  },
  {
    id: "pangkat",
    type: "pre",
    ask: "singkatan pangkat TNI dan Polri yang ditulis di depan nama (AKBP, Kombes Pol., Letkol Inf. ...)",
    example: '{"surface": "AKBP", "id": "Ajun Komisaris Besar Polisi", "en": "Adjunct Senior Commissioner of Police"}',
  },
];

// ---------------------------------------------------------------- arguments and files

function parseArgs(argv) {
  const args = { cmd: argv[2], config: "llm.json", out: "out", perCategory: 200, batch: 40, grounding: [], strict: false };
  for (let i = 3; i < argv.length; i++) {
    const [k, v] = [argv[i], argv[i + 1]];
    if (k === "--config") { args.config = v; i++; }
    else if (k === "--out") { args.out = v; i++; }
    else if (k === "--per-category") { args.perCategory = Number(v); i++; }
    else if (k === "--batch") { args.batch = Number(v); i++; }
    else if (k === "--grounding") { args.grounding.push(v); i++; }
    else if (k === "--strict") args.strict = true;
  }
  return args;
}

function readJsonl(file) {
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
}

function append(file, rows) {
  if (rows.length) fs.appendFileSync(file, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
}

function loadConfig(file) {
  const env = (v) => (typeof v === "string" && v.startsWith("$") ? process.env[v.slice(1)] || "" : v);
  const raw = JSON.parse(fs.readFileSync(file, "utf8"));
  const fix = (c) => Object.fromEntries(Object.entries(c).map(([k, v]) => [k, env(v)]));
  return { generator: fix(raw.generator), judges: raw.judges.map(fix) };
}

// Same key as titles.py: letters only, so "S. Kom." and "SKom" are one title.
const keyOf = (row) => `${row.type}:${(row.code || row.surface || "").replace(/[^A-Za-z]/g, "")}`;

// ---------------------------------------------------------------- LLM calls

async function chat(llm, messages, temperature, attempt = 1) {
  try {
    const res = await fetch(`${llm.baseUrl.replace(/\/+$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${llm.apiKey}` },
      body: JSON.stringify({ model: llm.model, messages, temperature, max_tokens: 8192 }),
      signal: AbortSignal.timeout(180_000),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      if ((res.status === 429 || res.status >= 500) && attempt <= 5) throw new Error(`HTTP ${res.status}`);
      throw Object.assign(new Error(`HTTP ${res.status}: ${body.slice(0, 200)}`), { fatal: true });
    }
    const data = await res.json();
    return data?.choices?.[0]?.message?.content || "";
  } catch (err) {
    if (err.fatal || attempt > 5) throw err;
    const wait = Math.min(30_000, 1000 * 2 ** attempt) + Math.random() * 500;
    console.warn(`  retry ${attempt} after ${err.message}, ${Math.round(wait)} ms`);
    await new Promise((r) => setTimeout(r, wait));
    return chat(llm, messages, temperature, attempt + 1);
  }
}

function parseArray(text) {
  const clean = text.replace(/```(?:json)?/g, "").trim();
  const start = clean.indexOf("[");
  const end = clean.lastIndexOf("]");
  if (start < 0 || end <= start) return [];
  try {
    const arr = JSON.parse(clean.slice(start, end + 1));
    return Array.isArray(arr) ? arr.filter((x) => x && typeof x === "object") : [];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------- step 1: generate

function groundingChunks(files, size = 12_000) {
  const text = files.map((f) => fs.readFileSync(f, "utf8")).join("\n\n");
  const chunks = [];
  for (let i = 0; i < text.length; i += size) chunks.push(text.slice(i, i + size));
  return chunks;
}

function generatorPrompt(cat, n, known, grounding, strict) {
  const rules = [
    "Kamu menyusun leksikon pembacaan gelar untuk TTS (text-to-speech) bahasa Indonesia.",
    `Tugas: daftar ${cat.ask}.`,
    `Format: JSON array saja, tanpa penjelasan. Contoh satu baris: ${cat.example}`,
    "Isi \"id\" dengan kepanjangan resmi dalam bahasa Indonesia seperti yang diucapkan pembawa acara.",
    "Isi \"en\" dengan nama resminya dalam bahasa Inggris. Untuk gelar luar negeri, id dan en sama.",
    "Tulis kepanjangan dengan huruf kapital di awal kata (Title Case), jangan ada singkatan huruf besar di dalamnya.",
    "Hanya yang benar-benar dipakai. Kalau ragu, jangan dimasukkan.",
    `Maksimal ${n} baris. Jangan ulangi yang sudah ada: ${known.slice(0, 150).join(", ") || "-"}.`,
  ];
  if (grounding) {
    rules.push(strict
      ? "Ambil HANYA gelar yang tertulis di dokumen berikut. Jangan menambah dari ingatan."
      : "Utamakan gelar yang tertulis di dokumen berikut.");
    rules.push(`Dokumen:\n"""\n${grounding}\n"""`);
  }
  return [{ role: "user", content: rules.join("\n") }];
}

async function generate(args, cfg) {
  fs.mkdirSync(args.out, { recursive: true });
  const file = path.join(args.out, "candidates.jsonl");
  const have = new Map(readJsonl(file).map((r) => [keyOf(r), r]));
  const chunks = args.grounding.length ? groundingChunks(args.grounding) : [null];

  for (const cat of CATEGORIES) {
    let stale = 0;
    for (let round = 0; ; round++) {
      const mine = [...have.values()].filter((r) => r.category === cat.id);
      if (mine.length >= args.perCategory || stale >= 3) break;
      const known = mine.map((r) => r.code || r.surface);
      const chunk = chunks[round % chunks.length];
      const reply = await chat(cfg.generator, generatorPrompt(cat, args.batch, known, chunk, args.strict), 0.9);
      const fresh = [];
      for (const item of parseArray(reply)) {
        const row = { type: cat.type, category: cat.id, ...item, source: cfg.generator.model };
        const name = row.code || row.surface;
        if (!name || !row.id || !/[A-Za-z]/.test(name)) continue;
        const k = keyOf(row);
        if (have.has(k)) continue;
        have.set(k, row);
        fresh.push(row);
      }
      append(file, fresh);
      stale = fresh.length ? 0 : stale + 1;
      console.log(`${cat.id.padEnd(16)} round ${round}: +${fresh.length} (${mine.length + fresh.length})`);
    }
  }
}

// ---------------------------------------------------------------- step 2: verify

function judgePrompt(rows) {
  const lines = rows.map((r, i) => {
    const what = r.type === "field" ? `kode bidang "${r.code}" (contoh: S.${r.code}.)`
      : r.type === "spesialis" ? `kode spesialis "Sp.${r.code}"`
      : r.type === "konsultan" ? `kode konsultan "Sp.PD-K${r.code}"`
      : r.type === "pre" ? `gelar/sebutan di depan nama "${r.surface}"`
      : `gelar di belakang nama "${r.surface}"`;
    return `${i + 1}. ${what}`;
  });
  return [{
    role: "user",
    content: [
      "Untuk setiap singkatan gelar berikut (konteks: penulisan gelar di Indonesia), jawab apakah singkatan itu benar-benar dipakai,",
      "dan apa kepanjangan resminya dalam bahasa Indonesia (id) dan bahasa Inggris (en).",
      "Kalau singkatan itu punya lebih dari satu arti yang lazim, isi ambiguous: true.",
      'Jawab JSON array saja, urut sesuai nomor: [{"n": 1, "real": true, "id": "...", "en": "...", "ambiguous": false}]',
      "",
      ...lines,
    ].join("\n"),
  }];
}

async function verify(args, cfg) {
  const candidates = readJsonl(path.join(args.out, "candidates.jsonl"));
  const file = path.join(args.out, "verdicts.jsonl");
  const done = new Set(readJsonl(file).map((v) => `${v.judge}|${v.key}`));

  for (const judge of cfg.judges) {
    const todo = candidates.filter((r) => !done.has(`${judge.name}|${keyOf(r)}`));
    console.log(`${judge.name}: ${todo.length} to check`);
    for (let i = 0; i < todo.length; i += 20) {
      const rows = todo.slice(i, i + 20);
      const answers = parseArray(await chat(judge, judgePrompt(rows), 0));
      const verdicts = rows.map((r, j) => {
        const a = answers.find((x) => Number(x.n) === j + 1) || {};
        return { judge: judge.name, key: keyOf(r), real: a.real === true, id: a.id || null, en: a.en || null,
                 ambiguous: a.ambiguous === true };
      });
      append(file, verdicts);
      console.log(`  ${Math.min(i + 20, todo.length)}/${todo.length}`);
    }
  }
}

// ---------------------------------------------------------------- step 3: export

const STOP = new Set(["dan", "and", "of", "in", "the", "di", "ilmu", "bidang"]);
function words(s) {
  return new Set(String(s || "").toLowerCase().normalize("NFD").replace(/[^a-z ]/g, " ").split(/\s+/)
    .filter((w) => w && !STOP.has(w)));
}
function agree(a, b) {
  const A = words(a), B = words(b);
  if (!A.size || !B.size) return false;
  const inter = [...A].filter((w) => B.has(w)).length;
  return inter / new Set([...A, ...B]).size >= 0.8;
}

function exportRows(args, cfg) {
  const candidates = readJsonl(path.join(args.out, "candidates.jsonl"));
  const verdicts = readJsonl(path.join(args.out, "verdicts.jsonl"));
  const judges = cfg ? cfg.judges.map((j) => j.name) : [...new Set(verdicts.map((v) => v.judge))];
  const byKey = new Map();
  for (const v of verdicts) byKey.set(`${v.judge}|${v.key}`, v);

  const accepted = [], review = [];
  for (const row of candidates) {
    const k = keyOf(row);
    const vs = judges.map((j) => byKey.get(`${j}|${k}`)).filter(Boolean);
    const reasons = [];
    if (vs.length < judges.length) reasons.push("not checked by every judge");
    if (vs.some((v) => !v.real)) reasons.push("a judge says it is not used");
    if (vs.some((v) => v.ambiguous)) reasons.push("ambiguous");
    if (vs.some((v) => !agree(row.id, v.id))) reasons.push("expansions disagree");
    if (/\b[A-Z]{2,}\b/.test(`${row.id} ${row.en || ""}`)) reasons.push("capitals inside the expansion");
    const out = { ...row, en: row.en || vs.find((v) => v.en)?.en || row.id, judges: vs };
    (reasons.length ? review : accepted).push(reasons.length ? { ...out, reasons } : out);
  }

  const write = (name, rows) => fs.writeFileSync(path.join(args.out, name),
    rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : ""));
  write("accepted.jsonl", accepted.map(({ judges: _, ...r }) => r));
  write("review.jsonl", review);
  console.log(`accepted ${accepted.length}, review ${review.length} (of ${candidates.length})`);
  const tally = {};
  for (const r of review) for (const why of r.reasons) tally[why] = (tally[why] || 0) + 1;
  console.log("review reasons:", tally);
}

// ---------------------------------------------------------------- main

const args = parseArgs(process.argv);
const needsLlm = ["generate", "verify", "all"].includes(args.cmd);
const cfg = needsLlm || fs.existsSync(args.config) ? loadConfig(args.config) : null;

if (args.cmd === "generate") await generate(args, cfg);
else if (args.cmd === "verify") await verify(args, cfg);
else if (args.cmd === "export") exportRows(args, cfg);
else if (args.cmd === "all") { await generate(args, cfg); await verify(args, cfg); exportRows(args, cfg); }
else {
  console.log("usage: node generate_title_dataset.mjs generate|verify|export|all --config llm.json [--out out]");
  process.exit(1);
}
