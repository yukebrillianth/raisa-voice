#!/usr/bin/env node
// Sentences for the text tagger (orchestrator/text_tagger.py): every word gets a tag.
//
//   language of a word:      ID, EN, JV, NAME (people, places, brands: no language of their own)
//   capitals (2+ letters):   WORD (Unesa), SPELL_ID (u-gé-ém), SPELL_EN (GPU)
//   digits and punctuation:  X, added here, never asked
//
// Step 1 (generate): one model writes sentences the way RAISA would say them, in Indonesian,
// English, Javanese and code-switched mixes, many with capitals and numbers.
// Step 2 (label): every judge tags every word blind, at temperature 0.
// Step 3 (export): a word's tag is the majority; capitals need every judge to agree; a sentence
// with any word nobody agrees on goes to review. The user's rules (orchestrator/acronym_data.csv,
// source "user-rule") override the judges for those tokens.
//
//   node generate_sentences.mjs generate --config llm.json --target 20000
//   node generate_sentences.mjs label --config llm.json
//   node generate_sentences.mjs export          # out/train.jsonl, out/review.jsonl
//   python train_tagger.py --data out/train.jsonl --out out/model
//
// llm.json: same shape as train/titles/llm.example.json, with 2-3 judges from different families.
// Node 18+, no dependencies; every step appends and resumes.

import fs from "node:fs";
import path from "node:path";

// Must match WORD in orchestrator/text_tagger.py.
const WORD = /[A-Za-zÀ-ÿ]+|\d+|[^\sA-Za-zÀ-ÿ\d]/gu;
const isCaps = (w) => /^[A-Z]{2,}$/.test(w);
const isAlpha = (w) => /^[A-Za-zÀ-ÿ]+$/.test(w);
const LANG_TAGS = new Set(["ID", "EN", "JV", "NAME"]);
const CAP_TAGS = new Set(["WORD", "SPELL_ID", "SPELL_EN"]);

const MODES = [
  ["id", 0.38, "bahasa Indonesia santai tapi sopan, seperti asisten robot pemandu"],
  ["en", 0.2, "English only, friendly assistant tone"],
  ["mix", 0.27, "bahasa Indonesia dengan sisipan kata/frasa Inggris yang lazim (code-switching anak muda/teknologi)"],
  ["jv", 0.08, "bahasa Jawa (ngoko atau krama ringan), boleh campur sedikit bahasa Indonesia"],
  ["en_id", 0.07, "English with a few Indonesian words or names mixed in"],
];
const DOMAINS = [
  "kampus dan pendaftaran mahasiswa (ITS, UGM, UNAIR, UTBK, LPDP, UKT, KRS)",
  "rumah sakit dan kesehatan (IGD, BPJS, RSUD, USG, ICU, dokter spesialis)",
  "layanan pemerintah dan dokumen (KTP, SIM, NPWP, SKCK, BPOM, Dukcapil)",
  "komputer, AI, dan gadget (GPU, CPU, RAM, SSD, USB, HDMI, API, LLM, RTX 5090, PDF)",
  "robotika dan IoT (ROS, LIDAR, UART, ESP32, PWM, TTS, ASR)",
  "transportasi dan kota (KRL, MRT, KAI, SPBU, PLN, Surabaya, Jogja)",
  "bank, belanja, dan pembayaran (BCA, QRIS, ATM, COD, OJK, Rp 150.000)",
  "pameran, galeri, dan acara kampus (jadwal, lantai, pukul 09.00-15.00, EXPO, HIMATEKTRO)",
  "perusahaan dan kerja (CEO, HRD, SOP, KPI, PT, CV, meeting, deadline)",
  "pariwisata dan kuliner Jawa Timur (Bromo, Malang, rawon, tiket, harga)",
];

function parseArgs(argv) {
  const a = { cmd: argv[2], config: "llm.json", out: "out", target: 20000, batch: 25, concurrency: 4 };
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

function parseJson(text, open = "[", close = "]") {
  const clean = text.replace(/```(?:json)?/g, "");
  const s = clean.indexOf(open), e = clean.lastIndexOf(close);
  if (s < 0 || e <= s) return null;
  try { return JSON.parse(clean.slice(s, e + 1)); } catch { return null; }
}

const pick = (list) => {
  let r = Math.random() * list.reduce((t, x) => t + x[1], 0);
  for (const x of list) if ((r -= x[1]) <= 0) return x;
  return list[0];
};

// ---------------------------------------------------------------- generate

async function generate(a, cfg) {
  fs.mkdirSync(a.out, { recursive: true });
  const file = path.join(a.out, "sentences.jsonl");
  const seen = new Set(readJsonl(file).map((r) => r.text));
  let made = seen.size, stale = 0, round = 0;

  async function worker() {
    while (made < a.target && stale < 20) {
      const [mode, , style] = pick(MODES);
      const domain = DOMAINS[round++ % DOMAINS.length];
      const prompt = [
        `Tulis ${a.batch} kalimat yang mungkin diucapkan RAISA, robot pemandu di kampus/galeri, kepada pengunjung.`,
        `Bahasa: ${style}. Topik: ${domain}.`,
        "Variasikan panjang (3-30 kata), bentuk (jawaban, pertanyaan, daftar, angka, jam, harga, rentang tahun),",
        "dan sertakan kata berhuruf kapital semua (singkatan, akronim, nama yang ditulis kapital) di sekitar 70% kalimat,",
        "termasuk yang jarang dan yang dibaca utuh sebagai kata. Jangan pakai markdown atau emoji.",
        'Jawab JSON array berisi string saja: ["kalimat 1", "kalimat 2"]',
      ].join("\n");
      try {
        const items = parseJson(await chat(cfg.generator, prompt, 1.0)) || [];
        const fresh = [];
        for (const t of items) {
          const text = typeof t === "string" ? t.trim() : "";
          if (text.length < 8 || text.length > 400 || seen.has(text)) continue;
          seen.add(text);
          fresh.push({ text, mode, domain, source: cfg.generator.model });
        }
        append(file, fresh);
        made += fresh.length;
        stale = fresh.length ? 0 : stale + 1;
        console.log(`${mode.padEnd(6)} +${fresh.length} -> ${made}/${a.target}`);
      } catch (err) {
        console.error(err.message);
        stale++;
      }
    }
  }
  await Promise.all(Array.from({ length: a.concurrency }, worker));
}

// ---------------------------------------------------------------- label

function labelPrompt(batch) {
  const lines = batch.map((r, i) => {
    const words = r.text.match(WORD).filter(isAlpha);
    return `${i + 1}. ${words.map((w, j) => `${j}:${w}`).join(" ")}`;
  });
  return [
    "Beri label untuk SETIAP kata di tiap kalimat (kata sudah dinomori).",
    "Kata huruf kapital semua (2 huruf atau lebih), menurut cara orang Indonesia mengucapkannya di kalimat itu:",
    "  WORD = dibaca sebagai kata (UNAIR, BPOM, PUSKESMAS, WIFI, NASA), SPELL_ID = dieja huruf Indonesia (ITS \"i-té-és\", KTP, WIB, HDMI),",
    "  SPELL_EN = dieja huruf Inggris (GPU, USB, PDF, API, AI, TV, VIP).",
    "  Di kalimat bahasa Inggris, akronim tetap diberi label cara bacanya dalam bahasa Inggris (biasanya SPELL_EN atau WORD).",
    "Kata lain: ID (bahasa Indonesia, termasuk serapan yang sudah lazim), EN (bahasa Inggris), JV (bahasa Jawa),",
    "  NAME (nama orang, tempat, merek, judul: tidak punya bahasa sendiri).",
    'Jawab JSON array, satu objek per kalimat sesuai nomor: [{"n": 1, "tags": ["ID", "EN", "SPELL_ID", ...]}]',
    "Panjang tags harus sama dengan jumlah kata di kalimat itu.",
    "",
    ...lines,
  ].join("\n");
}

async function label(a, cfg) {
  const rows = readJsonl(path.join(a.out, "sentences.jsonl"));
  const file = path.join(a.out, "labels.jsonl");
  const done = new Set(readJsonl(file).map((v) => `${v.judge}|${v.text}`));
  for (const judge of cfg.judges) {
    const todo = rows.filter((r) => !done.has(`${judge.name}|${r.text}`));
    console.log(`${judge.name}: ${todo.length} sentences to label`);
    const batches = [];
    for (let i = 0; i < todo.length; i += 15) batches.push(todo.slice(i, i + 15));
    let next = 0;
    async function worker() {
      while (next < batches.length) {
        const batch = batches[next++];
        try {
          const answers = parseJson(await chat(judge, labelPrompt(batch), 0)) || [];
          const out = batch.map((r, i) => {
            const x = answers.find((y) => Number(y.n) === i + 1);
            return { judge: judge.name, text: r.text, tags: Array.isArray(x?.tags) ? x.tags.map(String) : null };
          });
          append(file, out);
        } catch (err) {
          console.error(`  ${judge.name}: ${err.message}`);
        }
        if (next % 20 === 0) console.log(`  ${judge.name}: ${Math.min(next * 15, todo.length)}/${todo.length}`);
      }
    }
    await Promise.all(Array.from({ length: a.concurrency }, worker));
  }
}

// ---------------------------------------------------------------- export

function userRules() {
  const here = path.dirname(new URL(import.meta.url).pathname);
  const csv = path.resolve(here, "../../orchestrator/acronym_data.csv");
  const rules = new Map();
  if (!fs.existsSync(csv)) return rules;
  for (const line of fs.readFileSync(csv, "utf8").split("\n").slice(1)) {
    const [token, lab, source] = line.split(",");
    if (source === "user-rule") rules.set(token, lab);
  }
  return rules;
}

function exportRows(a, cfg) {
  const rows = readJsonl(path.join(a.out, "sentences.jsonl"));
  const labels = readJsonl(path.join(a.out, "labels.jsonl"));
  const judges = cfg ? cfg.judges.map((j) => j.name) : [...new Set(labels.map((l) => l.judge))];
  const by = new Map(labels.map((l) => [`${l.judge}|${l.text}`, l.tags]));
  const rules = userRules();
  const train = [], review = [];

  for (const r of rows) {
    const words = r.text.match(WORD) || [];
    const alpha = words.map((w, i) => [w, i]).filter(([w]) => isAlpha(w));
    const votes = judges.map((j) => by.get(`${j}|${r.text}`)).filter((t) => Array.isArray(t) && t.length === alpha.length);
    const why = [];
    if (votes.length < Math.min(2, judges.length)) why.push("too few usable labels");
    const tags = words.map((w) => (isAlpha(w) ? null : "X"));
    alpha.forEach(([w, i], k) => {
      if (why.length) return;
      const caps = isCaps(w);
      const options = votes.map((v) => String(v[k]).toUpperCase()).filter((t) => (caps ? CAP_TAGS : LANG_TAGS).has(t));
      if (caps && rules.has(w)) { tags[i] = rules.get(w); return; }
      const count = {};
      for (const t of options) count[t] = (count[t] || 0) + 1;
      const [best, n] = Object.entries(count).sort((x, y) => y[1] - x[1])[0] || [null, 0];
      if (caps ? n < votes.length : n <= votes.length / 2) why.push(`${w}: ${options.join("/") || "no tag"}`);
      tags[i] = best;
    });
    (why.length ? review : train).push(why.length ? { ...r, why } : { text: r.text, words, tags, mode: r.mode, domain: r.domain });
  }
  const write = (name, list) => fs.writeFileSync(path.join(a.out, name), list.map((x) => JSON.stringify(x)).join("\n") + (list.length ? "\n" : ""));
  write("train.jsonl", train);
  write("review.jsonl", review);
  console.log(`train ${train.length}, review ${review.length} (of ${rows.length})`);
}

const a = parseArgs(process.argv);
const cfg = ["generate", "label", "all"].includes(a.cmd) || fs.existsSync(a.config) ? loadConfig(a.config) : null;
if (a.cmd === "generate") await generate(a, cfg);
else if (a.cmd === "label") await label(a, cfg);
else if (a.cmd === "export") exportRows(a, cfg);
else if (a.cmd === "all") { await generate(a, cfg); await label(a, cfg); exportRows(a, cfg); }
else { console.log("usage: node generate_sentences.mjs generate|label|export|all --config llm.json"); process.exit(1); }
