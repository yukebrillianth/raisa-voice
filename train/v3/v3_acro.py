"""LoRA v3 acronym and word test set: 2 Indonesian carriers per acronym of the verified set, English carriers where
the reading depends on the language, the judge-holdout clips, and the se-/ke-/colok- words.

Carriers for the 22 acronyms of acro.py are acro.py's own (so the numbers compare with acro_report.md); the other
32 are new here. None is in any training transcript (v3_prep.py asserts it).
Targets (what a judge must hear) = acro.py's rule reading plus every accepted spelling of the acronym manifest.
"""
import json
import re

from acro import CARRIERS as ACRO_CARRIERS, cases as acro_cases, g2p

CARRIERS = {
    "AI": ["Robot ini memakai {} untuk memahami pertanyaan tamu.", "Teknologi {} sekarang dipakai di banyak kantor."],
    "BCA": ["Pembayaran lewat transfer {} juga diterima.", "Kartu debit {} bisa dipakai di kasir."],
    "BIMA": ["Robot {} siap mengantar Anda ke ruang rapat.", "Tolong panggil {} untuk membantu tamu ini."],
    "BNI": ["Rekening {} panitia tertera di brosur.", "Kantor cabang {} ada di seberang jalan."],
    "BRI": ["Silakan transfer ke rekening {} milik koperasi.", "Petugas {} sedang membuka stan di aula."],
    "BRIN": ["Peneliti {} akan berkunjung minggu depan.", "Kerja sama dengan {} sudah ditandatangani."],
    "BUMN": ["Banyak lulusan kami bekerja di {} sekarang.", "Perwakilan {} sudah hadir di ruang tamu."],
    "CPU": ["Suhu {} laptop ini cukup tinggi.", "Kipas {} di komputer itu perlu dibersihkan."],
    "EXPO": ["Pameran {} dibuka setiap akhir pekan.", "Tiket masuk {} bisa dibeli di pintu timur."],
    "FTEIC": ["Gedung {} ada di sebelah perpustakaan.", "Dekan {} akan membuka acara ini."],
    "HD": ["Kamera ini merekam video {} dengan jernih.", "Layar {} di ruang rapat baru dipasang."],
    "HIMATEKTRO": ["Pengurus {} sedang rapat di sekretariat.", "Acara ini diselenggarakan oleh {} tahun ini."],
    "IPK": ["Nilai {} Anda bisa dilihat di portal akademik.", "Beasiswa itu memperhatikan {} dan prestasi lomba."],
    "IRIS": ["Robot tim {} sedang dipamerkan di lobi.", "Anggota {} sedang memperbaiki robot beroda."],
    "ITB": ["Rombongan dari {} sudah tiba di gerbang.", "Dosen tamu dari {} akan mengisi kuliah umum."],
    "KRS": ["Pengisian {} dibuka mulai besok pagi.", "Jangan lupa mencetak {} sebelum bertemu dosen wali."],
    "KTM": ["Tunjukkan {} Anda kepada petugas perpustakaan.", "Pengambilan {} baru dilayani di gedung rektorat."],
    "LAN": ["Sambungkan komputer ke kabel {} di bawah meja.", "Jaringan {} di ruang ini sedang diperbaiki."],
    "LED": ["Lampu {} di panggung berganti warna.", "Layar {} di aula menampilkan jadwal acara."],
    "LUNA": ["Robot {} sedang mengantar tamu ke lantai dua.", "Perkenalkan, ini {}, robot pemandu kami."],
    "MAGE": ["Lomba {} tahun ini diikuti banyak sekolah.", "Pendaftaran {} ditutup akhir bulan ini."],
    "NABBIT": ["Robot {} sedang mengisi daya di pojok ruangan.", "Tim {} memenangkan lomba robot kemarin."],
    "NPWP": ["Tolong lampirkan fotokopi {} Anda.", "Nomor {} diperlukan untuk pencairan honor."],
    "PC": ["Silakan gunakan {} di ruang baca.", "Aplikasi ini hanya berjalan di {} kantor."],
    "PKM": ["Proposal {} dikumpulkan ke bagian kemahasiswaan.", "Tim {} kami lolos ke tingkat nasional."],
    "RAM": ["Kapasitas {} laptop ini masih kurang.", "Tambahkan {} supaya komputernya lebih cepat."],
    "SIM": ["Bawa {} Anda kalau ingin meminjam mobil kampus.", "Pembuatan {} bisa dilakukan di kantor polisi terdekat."],
    "SKEM": ["Kegiatan ini memberi nilai {} untuk peserta.", "Laporan {} harus diunggah sebelum wisuda."],
    "SPARK": ["Kompetisi {} digelar di aula utama.", "Peserta {} diminta berkumpul di lobi."],
    "SSD": ["Laptop ini sudah memakai {} yang cepat.", "Data penelitian disimpan di {} cadangan."],
    "UGM": ["Mahasiswa dari {} sedang berkunjung ke laboratorium.", "Tim robot {} juga ikut lomba ini."],
    "UI": ["Delegasi dari {} akan tiba siang ini.", "Dosen {} itu menjadi juri lomba."],
}
# se-/ke- words lose an e and colok-words get a k (speech_text.FIXED); served_ab.SEWORDS carriers for the first three.
WORDS = {"seekor": "Ada {} kucing di depan lobi.", "seenaknya": "Jangan parkir {} di depan pintu.",
         "keenam": "Ruangannya ada di lantai {}.", "colokkan": "Silakan {} kabel laptop ke stopkontak di dinding."}
# The respelling served for v2; scored on v3 too, to decide whether the lexicon entry can go.
OVERRIDES = {"seenaknya": "se-enaknya", "colokkan": "cholokkan"}
# acro.py carriers that turned out to be in the v3 training text, replaced by a new sentence.
SWAP = {"Selamat datang di kampus {} Surabaya.": "Kantor rektorat {} buka sampai sore hari."}
MANIFEST = "/yuke/raisa-voice-data-v3/voxcpm2_lora_v3_acronym.jsonl"


def phones(spelling: str) -> list[str]:
    """Accepted spelling ("eichdiemai", "kyuris") -> phones; "ch"/"tch" is one sound, as gemini_judge.phones."""
    return g2p(re.sub(r"t?ch", "c", spelling))


def cases(hold_acr: list[dict], manifest: str = MANIFEST) -> list[dict]:
    acc = {}
    for r in map(json.loads, open(manifest, encoding="utf-8")):
        for t, a in r["expect"].items():
            acc.setdefault((t, r["lang"]), set()).update(a)
    out = []

    def add(kind, token, lang, n, orig, left, target, right, **extra):
        targets = [target] if target else []
        for spelling in sorted(acc.get((token, lang), ())):
            if phones(spelling) not in targets:
                targets.append(phones(spelling))
        out.append({"id": f"{kind}_{token}_{lang}{n}", "kind": kind, "token": token, "lang": lang, "orig": orig,
                    "left": left, "targets": targets, "right": right, **extra})

    for acronym, n, lang, forms, left, target, right in acro_cases():
        template = forms["raw"].replace(acronym, "{}", 1)
        if template in SWAP:
            forms, (left, right) = {"raw": SWAP[template].format(acronym)}, map(g2p, SWAP[template].split("{}"))
        add("acro" if lang == "id" else "en", acronym, lang, n, forms["raw"], left, target, right)
    for acronym, templates in CARRIERS.items():
        assert acronym not in ACRO_CARRIERS
        for n, t in enumerate(templates):
            left, right = t.split("{}")
            add("new", acronym, "id", n, t.format(acronym), g2p(left), None, g2p(right))
    for r in hold_acr:
        (token,) = r["expect"]
        left, right = re.split(rf"\b{token}\b", r["text"], maxsplit=1)
        add("hold", token, r["lang"], 0, r["text"], g2p(left), None, g2p(right), audio=r["audio"])
    for word, t in WORDS.items():
        left, right = t.split("{}")
        add("word", word, "id", 0, t.format(word), g2p(left), g2p(word), g2p(right))
        if word in OVERRIDES:
            add("word", word, "id", 1, t.format(word), g2p(left), g2p(word), g2p(right),
                v3=t.format(OVERRIDES[word]), only="v3")
    return out


if __name__ == "__main__":
    assert phones("eichdiemai") == list("eicdiemai") and phones("kyuris") == list("kyuris")
    print(len(CARRIERS) * 2, "new carriers")


# --- é-hyphen spelling test (the user heard "wé-i-bé" right on the served v2): raw vs Indonesian letter names with the
# taling e marked and joined by "-", 1 carrier x 8 samples per form, on the served v2 and on v3. NIK, SKCK are unseen.
ID_LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      "a bé cé dé é éf gé ha i jé ka él ém én o pé ki ér és té u vé wé éks yé zét".split()))
SPELL = {"WIB": "Acaranya mulai pukul sembilan {} di aula utama.", "ITS": "Robot ini dibuat oleh mahasiswa {} tahun lalu.",
         "HDMI": "Sambungkan laptop pakai kabel {} di meja.", "KTP": "Tolong siapkan {} Anda untuk pendaftaran.",
         "UMKM": "Pameran {} dibuka mulai hari ini.", "BPJS": "Kartu {} bisa dipakai di klinik ini.",
         "PKM": "Proposal {} dikumpulkan ke bagian kemahasiswaan.", "NIK": "Silakan tulis {} Anda di formulir pendaftaran.",
         "SKCK": "Surat {} bisa diurus di kantor polisi terdekat."}
SPELL_WORDS = {"seekor": ("Ada {} kucing di depan lobi.", ["seekor", "se-ékor"]),
               "keenam": ("Ruangannya ada di lantai {}.", ["keenam", "ke-enam", "ke-ènam"])}


def e_hyphen(acronym: str) -> str:
    return "-".join(ID_LETTERS[c] for c in acronym)


def spell_cases() -> list[dict]:
    out = []
    for token, t in SPELL.items():
        left, right = t.split("{}")
        forms = {"raw": token, "ehyph": e_hyphen(token)} | ({"spaced": "we i be"} if token == "WIB" else {})
        for form, written in forms.items():
            text = t.format(written)
            out.append({"id": f"spell_{token}_{form}", "kind": "spell", "token": token, "form": form, "lang": "id", "orig": text,
                        "v2": text, "v3": text, "left": g2p(left), "targets": [g2p(e_hyphen(token).replace("-", " "))], "right": g2p(right)})
    for word, (t, forms) in SPELL_WORDS.items():
        left, right = t.split("{}")
        for n, written in enumerate(forms):
            text = t.format(written)
            out.append({"id": f"word_{word}_f{n}", "kind": "word", "token": word, "form": written, "lang": "id", "orig": text,
                        "v2": text, "v3": text, "left": g2p(left), "targets": [g2p(word)], "right": g2p(right)})
    return out


assert e_hyphen("WIB") == "wé-i-bé" and e_hyphen("HDMI") == "ha-dé-ém-i" and e_hyphen("BPJS") == "bé-pé-jé-és"
assert g2p("wé i bé") == list("weibe")
