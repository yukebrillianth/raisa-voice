#!/usr/bin/env node
// More labelled ALL-CAPS tokens for orchestrator/acronym_clf.py, checked by several LLMs.
//
// Step 1 (generate) asks one model, at high temperature, for tokens across many domains.
// Step 2 (verify) asks each judge for the label of every token at temperature 0. The judge never
// sees the generator's label, and is also asked whether the token has a second common reading.
// Step 3 (export) keeps a token only when every judge gives the generator's label and nobody
// calls it ambiguous. The rest goes to review.csv for a human.
//
//   node generate_acronym_dataset.mjs generate --config llm.json --target 3000
//   node generate_acronym_dataset.mjs verify   --config llm.json
//   node generate_acronym_dataset.mjs export
//   cd ../../orchestrator && python acronym_clf.py train --extra ../train/acronyms/out/accepted.csv \
//       --report ../train/acronyms/out
//
// llm.json has the same shape as train/titles/llm.example.json. Use judges from different model
// families, and gemini-4-carbon rather than hydrogen (CLAUDE.md section 5: credit is limited).
// Node 18+, no dependencies; every step appends and skips finished work.

import fs from "node:fs";
import path from "node:path";

const LABELS = new Set(["WORD", "SPELL_ID", "SPELL_EN"]);
const TOKEN = /^[A-Z]{2,14}$/;

const DOMAINS = [
  ["kampus", "universitas, politeknik, fakultas, himpunan, istilah akademik Indonesia (UNAIR, ITS, UGM, FTEIC, UTBK, LPDP, BEM, POLBAN)"],
  ["medis", "rumah sakit, klinik, penyakit, alat medis, singkatan kesehatan Indonesia (RSUD, IGD, ICU, THT, USG, BPJS, PUSKESMAS, ISPA)"],
  ["pemerintahan", "kementerian, lembaga, BUMN, dinas, dokumen, hukum Indonesia (KEMENKES, BPOM, BMKG, KTP, NPWP, SKCK, KPK)"],
  ["bisnis", "telekomunikasi, bank, fintech, pajak, ritel, perusahaan (INDOSAT, BCA, OJK, PPN, UMKM, CEO, HRD, SOP)"],
  ["nama", "kota, kabupaten, pulau, nama orang, robot, merek, kata umum yang ditulis kapital di banner (SURABAYA, JOGJA, RAISA, GARUDA, HARGA)"],
  ["hardware", "hardware PC, server, layar, port, storage, elektronik (GPU, CPU, NVME, SSD, VGA, USB, OLED, PSU, FPGA, BIOS)"],
  ["software", "AI, machine learning, jaringan, cloud, format file, protokol (LLM, API, SDK, HTTP, DNS, VPN, JSON, YAML, CUDA, LORA)"],
  ["robotika", "robotika, embedded, sensor, audio/video, IoT (ROS, LIDAR, IMU, UART, GPIO, PWM, RTSP, TTS, ASR, DSP, RADAR)"],
  ["transport", "transportasi, penerbangan, pelabuhan, energi, infrastruktur Indonesia (KAI, KRL, MRT, SPBU, PLN, PDAM, DAMRI)"],
  ["jebakan", "pasangan yang sering salah: dibaca utuh (UNAIR, BPOM, BRIN, OLED, WIFI, NASA, UNESCO) vs dieja Indonesia (UGM, BMKG, RSUD, KTP, WIB) vs dieja Inggris (GPU, PDF, API, CPU, VIP)"],
];

const RULES = `Kelas, untuk cara orang Indonesia mengucapkan token itu di dalam kalimat bahasa Indonesia:
- WORD: dibaca sebagai satu kata (RAISA, UNAIR, BPOM, PUSKESMAS, WIFI, OLED, NASA, JSON, LINUX).
- SPELL_ID: dieja huruf per huruf dengan nama huruf Indonesia (ITS "i-té-és", UGM, KTP, WIB, RSUD, BMKG, HDMI, ATM).
- SPELL_EN: dieja huruf per huruf dengan nama huruf Inggris (GPU "ji-pi-yu", CPU, PDF, API, USB, SSD, AI, LED, VIP, TV).`;

function parseArgs(argv) {
  const a = { cmd: argv[2], config: "llm.json", out: "out", target: 3000, batch: 80, concurrency: 3 };
  for (let i = 3; i < argv.length; i++) {
    const [k, v] = [argv[i], argv[i + 1]];
    if (k === "--config") { a.config = v; i++; }
    else if (k === "--out") { a.out = v; i++; }
    else if (k === "--target") { a.target = Number(v); i++; }
    else if (k === "--batch") { a.batch = Number(v); i++; }
    else if (k === "--concurrency") { a.concurrency = Number(v); i++; }
  }
  return a;
}

const readJsonl = (f) => (fs.existsSync(f) ? fs.readFileSync(f, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)) : []);
const append = (f, rows) => rows.length && fs.appendFileSync(f, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");

function loadConfig(file) {
  const env = (v) => (typeof v === "string" && v.startsWith("$") ? process.env[v.slice(1)] || "" : v);
  const fix = (c) => Object.fromEntries(Object.entries(c).map(([k, v]) => [k, env(v)]));
  const raw = JSON.parse(fs.readFileSync(file, "utf8"));
  return { generator: fix(raw.generator), judges: raw.judges.map(fix) };
}

async function chat(llm, content, temperature, attempt = 1) {
  try {
    const res = await fetch(`${llm.baseUrl.replace(/\/+$/, "")}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${llm.apiKey}` },
      body: JSON.stringify({ model: llm.model, messages: [{ role: "user", content }], temperature, max_tokens: 8192 }),
      signal: AbortSignal.timeout(180_000),
    });
    if (!res.ok) {
      if ((res.status === 429 || res.status >= 500) && attempt <= 5) throw new Error(`HTTP ${res.status}`);
      throw Object.assign(new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`), { fatal: true });
    }
    return (await res.json())?.choices?.[0]?.message?.content || "";
  } catch (err) {
    if (err.fatal || attempt > 5) throw err;
    const wait = Math.min(30_000, 1000 * 2 ** attempt) + Math.random() * 500;
    console.warn(`  retry ${attempt} after ${err.message}`);
    await new Promise((r) => setTimeout(r, wait));
    return chat(llm, content, temperature, attempt + 1);
  }
}

function parseArray(text) {
  const clean = text.replace(/```(?:json)?/g, "");
  const s = clean.indexOf("["), e = clean.lastIndexOf("]");
  if (s < 0 || e <= s) return [];
  try {
    const arr = JSON.parse(clean.slice(s, e + 1));
    return Array.isArray(arr) ? arr : [];
  } catch {
    // a reply cut off mid-array still gives its complete items
    return [...clean.matchAll(/"token"\s*:\s*"([A-Za-z]{2,14})"\s*,\s*"label"\s*:\s*"(WORD|SPELL_ID|SPELL_EN)"/g)]
      .map((m) => ({ token: m[1], label: m[2] }));
  }
}

// ---------------------------------------------------------------- generate

async function generate(a, cfg) {
  fs.mkdirSync(a.out, { recursive: true });
  const file = path.join(a.out, "candidates.jsonl");
  const seen = new Map(readJsonl(file).map((r) => [r.token, r]));
  const curated = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../../orchestrator/acronym_data.csv");
  const known = fs.existsSync(curated) ? fs.readFileSync(curated, "utf8").split("\n").slice(1).map((l) => l.split(",")[0]) : [];
  for (const t of known) if (t && !seen.has(t)) seen.set(t, { token: t, curated: true });

  let round = 0, stale = 0;
  const counts = () => {
    const c = { WORD: 0, SPELL_ID: 0, SPELL_EN: 0 };
    for (const r of seen.values()) if (!r.curated) c[r.label]++;
    return c;
  };
  const generated = () => [...seen.values()].filter((r) => !r.curated).length;

  async function worker() {
    while (generated() < a.target && stale < 15) {
      const [domain, desc] = DOMAINS[round++ % DOMAINS.length];
      const c = counts();
      const focus = Object.entries(c).sort((x, y) => x[1] - y[1])[0][0];
      const avoid = [...seen.keys()].sort(() => Math.random() - 0.5).slice(0, 80).join(", ");
      const prompt = [
        "Kamu menyusun dataset untuk TTS (text-to-speech) bahasa Indonesia: cara membaca token huruf kapital.",
        RULES,
        `Domain: ${desc}.`,
        `Buat ${a.batch} token nyata (hanya A-Z, 2-14 huruf) dari domain ini, sekitar 60% berlabel ${focus}.`,
        "Lewati token yang punya dua cara baca yang sama-sama lazim (mis. KB: keluarga berencana / kilobyte).",
        `Jangan ulangi: ${avoid}.`,
        'Jawab JSON array saja: [{"token": "RSUD", "label": "SPELL_ID"}]',
      ].join("\n");
      try {
        const items = parseArray(await chat(cfg.generator, prompt, 0.9));
        const fresh = [];
        for (const it of items) {
          const token = String(it.token || "").trim().toUpperCase();
          const label = String(it.label || "").trim().toUpperCase();
          if (!TOKEN.test(token) || !LABELS.has(label) || seen.has(token)) continue;
          const row = { token, label, domain, source: cfg.generator.model };
          seen.set(token, row);
          fresh.push(row);
        }
        append(file, fresh);
        stale = fresh.length ? 0 : stale + 1;
        const n = counts();
        console.log(`${domain.padEnd(12)} +${fresh.length} -> ${generated()}/${a.target} (W ${n.WORD} ID ${n.SPELL_ID} EN ${n.SPELL_EN})`);
      } catch (err) {
        console.error(`${domain}: ${err.message}`);
        stale++;
      }
    }
  }
  await Promise.all(Array.from({ length: a.concurrency }, worker));
}

// ---------------------------------------------------------------- verify

async function verify(a, cfg) {
  const rows = readJsonl(path.join(a.out, "candidates.jsonl"));
  const file = path.join(a.out, "verdicts.jsonl");
  const done = new Set(readJsonl(file).map((v) => `${v.judge}|${v.token}`));
  for (const judge of cfg.judges) {
    const todo = rows.filter((r) => !done.has(`${judge.name}|${r.token}`));
    console.log(`${judge.name}: ${todo.length} to check`);
    for (let i = 0; i < todo.length; i += 40) {
      const chunk = todo.slice(i, i + 40);
      const prompt = [
        RULES,
        "Untuk setiap token berikut, beri label kelasnya. Kalau token punya dua cara baca yang sama-sama lazim",
        "di Indonesia, atau kamu tidak tahu token itu, isi ambiguous: true.",
        'Jawab JSON array saja: [{"token": "...", "label": "...", "ambiguous": false}]',
        "",
        chunk.map((r) => r.token).join(", "),
      ].join("\n");
      const answers = new Map(parseArray(await chat(judge, prompt, 0)).map((x) => [String(x.token || "").toUpperCase(), x]));
      append(file, chunk.map((r) => {
        const x = answers.get(r.token) || {};
        return { judge: judge.name, token: r.token, label: String(x.label || "").toUpperCase() || null, ambiguous: x.ambiguous === true };
      }));
      console.log(`  ${Math.min(i + 40, todo.length)}/${todo.length}`);
    }
  }
}

// ---------------------------------------------------------------- export

function exportRows(a, cfg) {
  const rows = readJsonl(path.join(a.out, "candidates.jsonl"));
  const verdicts = readJsonl(path.join(a.out, "verdicts.jsonl"));
  const judges = cfg ? cfg.judges.map((j) => j.name) : [...new Set(verdicts.map((v) => v.judge))];
  const by = new Map(verdicts.map((v) => [`${v.judge}|${v.token}`, v]));
  const accepted = [], review = [];
  for (const r of rows) {
    const vs = judges.map((j) => by.get(`${j}|${r.token}`)).filter(Boolean);
    const why = [];
    if (vs.length < judges.length) why.push("not checked by every judge");
    if (vs.some((v) => v.ambiguous)) why.push("ambiguous");
    if (vs.some((v) => v.label !== r.label)) why.push(`labels ${r.label}/${vs.map((v) => v.label).join("/")}`);
    (why.length ? review : accepted).push({ ...r, why: why.join("; ") });
  }
  const csv = (list, extra) => ["token,label,source" + (extra ? ",why" : ""),
    ...list.map((r) => `${r.token},${r.label},${r.source}` + (extra ? `,"${r.why}"` : ""))].join("\n") + "\n";
  fs.writeFileSync(path.join(a.out, "accepted.csv"), csv(accepted, false));
  fs.writeFileSync(path.join(a.out, "review.csv"), csv(review, true));
  console.log(`accepted ${accepted.length}, review ${review.length} (of ${rows.length})`);
}

const a = parseArgs(process.argv);
const cfg = ["generate", "verify", "all"].includes(a.cmd) || fs.existsSync(a.config) ? loadConfig(a.config) : null;
if (a.cmd === "generate") await generate(a, cfg);
else if (a.cmd === "verify") await verify(a, cfg);
else if (a.cmd === "export") exportRows(a, cfg);
else if (a.cmd === "all") { await generate(a, cfg); await verify(a, cfg); exportRows(a, cfg); }
else { console.log("usage: node generate_acronym_dataset.mjs generate|verify|export|all --config llm.json"); process.exit(1); }
