"""Acronym reading test set and the judge's text side: expected phones, alignment, the acronym's region.

Phones are written as Indonesian spelling ("j i p i y u", "h a d e m i"), so a heard reading can be
read off directly. Vowel length, e/schwa and v/f are merged. Phones are kept as heard ("ha de em i" is
"h a d e e m i"); grading compares them with repeats collapsed.
"""
import re

# The user's reading rules: English letter names always (EN), sentence-language letters (LANG), words (WORD).
EN = {"USB": "yu es bi", "TV": "ti vi", "GPU": "ji pi yu", "GB": "ji bi", "API": "ei pi ai", "QR": "kyu ar",
      "NFC": "en ef si", "VIP": "vi ai pi", "ID": "ai di", "IT": "ai ti", "PDF": "pidief", "QRIS": "kyuris"}
LANG = {"WIB": "we i be", "WIT": "we i te", "ITS": "i te es", "HDMI": "ha de em i", "KTP": "ka te pe", "UMKM": "u em ka em",
        "BPJS": "be pe je es"}
WORD = {"WITA": "Wita", "UNAIR": "Unair", "RAISA": "Raisa"}
EN_LETTERS = {"HDMI": "H D M I", "ITS": "I T S"}  # what the phonetic form writes in an English sentence
EN_LETTER_PHONES = {"H": "e i c", "D": "d i", "M": "e m", "I": "a i", "T": "t i", "S": "e s"}
# Second round: other ways to write the readings that no model got right as "we i be" / "ha de em i"
# (the lone "i" is read as English "ai"), and API, whose "ei" comes out as "e".
SPELLINGS = {"WIB": ["we i be", "weibe", "wei be", "we ih be", "wé i bé", "we-i-be", "W I B", "WIB"],
             "HDMI": ["ha de em i", "ha de em ih", "hadeemi", "ha de emi", "ha-de-em-i", "H D M I", "HDMI"],
             "API": ["ei pi ai", "e pi ai", "A P I", "API"]}

CARRIERS = {
    "USB": ["Kabel {} ada di laci meja depan.", "Silakan pakai port {} di samping layar."],
    "TV": ["Acaranya disiarkan lewat {} kampus nanti malam.", "Remote {} ada di atas meja."],
    "GPU": ["Server ini memakai {} generasi terbaru.", "Harga {} sekarang sedang naik."],
    "GB": ["Kapasitas memorinya enam belas {} saja.", "Kuota internetnya sepuluh {} per bulan."],
    "API": ["Dokumentasi {} bisa dibaca di situs kami.", "Aplikasi ini memakai {} dari pihak ketiga."],
    "QR": ["Silakan pindai kode {} di meja depan.", "Kode {} itu ada di balik tiket Anda."],
    "NFC": ["Tempelkan kartu ke pembaca {} di pintu.", "Ponsel Anda harus punya fitur {} dulu."],
    "VIP": ["Ruang {} ada di lantai dua.", "Tamu {} masuk lewat pintu samping."],
    "ID": ["Tolong tunjukkan kartu {} Anda di sini.", "Nomor {} Anda tertulis di tiket."],
    "IT": ["Tim {} sedang memperbaiki jaringan.", "Silakan hubungi bagian {} di lantai tiga."],
    "PDF": ["Berkasnya dikirim dalam bentuk {} lewat email.", "Silakan unduh brosur {} dari situs kami."],
    "QRIS": ["Pembayaran bisa pakai {} di kasir.", "Silakan pindai {} untuk membayar."],
    "WIB": ["Acaranya mulai pukul sembilan {} di aula utama.", "Pintu ditutup pukul lima sore {} hari ini."],
    "WIT": ["Acaranya mulai pukul sembilan {} di aula utama.", "Pintu ditutup pukul lima sore {} hari ini."],
    "ITS": ["Selamat datang di kampus {} Surabaya.", "Robot ini dibuat oleh mahasiswa {} tahun lalu."],
    "HDMI": ["Sambungkan laptop pakai kabel {} di meja.", "Port {} ada di belakang layar."],
    "KTP": ["Tolong siapkan {} Anda untuk pendaftaran.", "Fotokopi {} dikumpulkan di meja depan."],
    "UMKM": ["Pameran {} dibuka mulai hari ini.", "Produk {} dijual di lantai satu."],
    "BPJS": ["Kartu {} bisa dipakai di klinik ini.", "Silakan daftar {} di loket dua."],
    "WITA": ["Rapat dimulai pukul dua siang {} di ruang dua.", "Pesawatnya tiba pukul tujuh malam {} nanti."],
    "UNAIR": ["Tim dari {} sudah tiba di lobi.", "Beliau adalah dosen {} di Surabaya."],
    "RAISA": ["Halo, saya {}, robot pelayan di sini.", "Nama saya {}, senang bertemu Anda."],
}
# English carriers: (sentence, phones before the acronym, phones after it), phones in this file's spelling.
EN_CARRIERS = {
    "HDMI": [("Please plug the {} cable into the screen.", "p l i s p l a g d e", "k e i b e l i n t u d e s k r i n"),
             ("The {} port is behind the monitor.", "d e", "p o r t i s b i h a i n d e m o n i t e r")],
    "ITS": [("Welcome to {}, the campus opens at seven.", "w e l k a m t u", "d e k a m p e s o u p e n s a t s e f e n"),
            ("This robot was built by students of {} last year.", "d i s r o u b o t w a s b i l t b a i s t u d e n t s o f", "l a s t y i r")],
    "QR": [("Please scan the {} code on the table.", "p l i s s k a n d e", "k o u d o n d e t e i b e l"),
           ("The {} code is printed on your ticket.", "d e", "k o u d i s p r i n t e d o n y o r t i k e t")],
    "API": [("You can read the {} guide on our website.", "y u k a n r i d d e", "g a i d o n a u e r w e b s a i t"),
            ("This app uses an {} from our partner.", "d i s a p y u s e s a n", "f r o m a u e r p a r t n e r")],
}

VOWELS = set("aiueo")
RECOGNISED = {"tʃ": "c", "dʒ": "j", "j": "y", "ŋ": "ng", "ɲ": "ny", "ʃ": "sy", "ʒ": "sy", "x": "kh", "ɡ": "g",
              "v": "f", "ɹ": "r", "ɾ": "r", "θ": "t", "ð": "d", "ɪ": "i", "ɛ": "e", "ə": "e", "ɚ": "e r",
              "ɜ": "e", "æ": "a", "ɐ": "a", "ʌ": "a", "ɑ": "a", "ɔ": "o", "ʊ": "u", "ʉ": "u", "ɒ": "o",
              "aɪ": "a i", "eɪ": "e i", "aʊ": "a u", "oʊ": "o u", "əʊ": "o u", "ɔɪ": "o i", "ɟ": "j", "ɖ": "d"}


def collapse(phones: list[str]) -> list[str]:
    return [p for i, p in enumerate(phones) if i == 0 or p != phones[i - 1]]


def heard(tokens: list[str]) -> list[str]:
    """Recogniser tokens (espeak IPA) in this file's spelling."""
    out, glided = [], False
    for t in tokens:
        t = re.sub(r"[ːʲʰˑ\[\]]", "", t)
        new = RECOGNISED.get(t, t).split() if t and t != "ʔ" else []
        if glided and new == out[-1:]:  # "aɪ" then "i": the diphthong's own end held on, not a new syllable
            glided = False
            continue
        out, glided = out + new, len(new) == 2 and new[1] in VOWELS
    return out


def g2p(text: str) -> list[str]:
    """Phones of Indonesian spelling (also of the phonetic acronym forms, which are written in it)."""
    text = text.lower().replace("remote", "remot").replace("é", "e")  # "remote" is said without its e
    text = re.sub(r"[^a-z ]", " ", text)
    for a, b in (("ng", " ŋ "), ("ny", " ɲ "), ("sy", " ʃ "), ("kh", " x "), ("x", "ks"), ("q", "k"), ("v", "f")):
        text = text.replace(a, b)
    out = []
    for token in text.split():
        out += [{"ŋ": "ng", "ɲ": "ny", "ʃ": "sy", "x": "kh"}[token]] if token in "ŋɲʃx" else list(token)
    return out


GLIDES = VOWELS | {"y", "w"}  # "kyu" and "kiu", "we" and "ue" are the same sounds
MERGE = [set("ei"), set("ouw")]  # "ei" heard as "e", "we i" as "wi": neighbours of one colour run together


def cost(a: str, b: str) -> float:
    return 0.0 if a == b else 0.5 if a in GLIDES and b in GLIDES else 1.0


def skip(seq: list[str], i: int) -> float:
    """Cost of a phone missing or extra: half when it is a vowel next to one of its own colour."""
    near = seq[max(0, i - 1): i] + seq[i + 1: i + 2]
    return 0.5 if any(seq[i] in m and n in m for m in MERGE for n in near) else 1.0


def distance(a: list[str], b: list[str], ends: bool = False):
    """Edit distance; with `ends`, the distance of `a` to every prefix of `b` (last DP row)."""
    row = [0.0]
    for j in range(len(b)):
        row.append(row[-1] + skip(b, j))
    for i, x in enumerate(a):
        prev, row[0] = row[0], row[0] + skip(a, i)
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + skip(a, i), row[j - 1] + skip(b, j - 1), prev + cost(x, y))
    return row if ends else row[-1]


def grade(got: list[str], target: list[str], left: list[str], right: list[str], lang: str = "id") -> str:
    """exact, near (one vowel-colour slip: "weibi", "epiai", "kiuar") or wrong (any consonant, or "ai" for "i").

    A phone shared with the neighbouring word is heard once ("lewat ti vi" -> "ifi"), so the reading may
    lack it. In English the final r may be silent and the letter H may start with an h.
    """
    # Doubled consonants are one sound; doubled vowels are two syllables ("de em"), kept so "hadimi" is not "ha de em i".
    squeeze = lambda ps: [p for i, p in enumerate(ps) if i == 0 or p != ps[i - 1] or p in VOWELS]
    got, target = squeeze(got), squeeze(target)
    targets = [target]
    targets += [target[1:]] if left and left[-1] == target[0] else []
    targets += [target[:-1]] if right and right[0] == target[-1] else []
    if lang == "en":
        targets += [t[:-1] for t in targets if t[-1] == "r"] + [["h"] + t for t in targets if t[:3] == ["e", "i", "c"]]
        targets += [t[1:] for t in targets if t[:2] == ["e", "i"]]  # "the API": the e runs into "the"
    d = min(distance(got, t) for t in targets)
    return "exact" if d == 0 else "near" if d <= 0.5 else "wrong"


def region(observed: list[str], left: list[str], target: list[str], right: list[str]) -> list[str]:
    """The part of `observed` that is neither the sentence before the acronym nor the one after it."""
    before = distance(left, observed, ends=True)                    # left vs observed[:i]
    after = distance(right[::-1], observed[::-1], ends=True)[::-1]  # right vs observed[j:]
    spans = [(i, j) for i in range(len(observed) + 1) for j in range(i, len(observed) + 1)]
    least = min(before[i] + after[j] for i, j in spans)
    # Among the cuts that fit the sentence equally well, the one closest to the rule reading.
    i, j = min((s for s in spans if before[s[0]] + after[s[1]] == least), key=lambda s: distance(observed[s[0]:s[1]], target))
    return observed[i:j]


def cases(spellings: bool = False):
    """Every (acronym, carrier index, language, {form: sentence}, left, target, right phones).

    Forms are "raw" and "phon" (the rule reading written out), or with `spellings` each SPELLINGS entry.
    """
    readings = {**EN, **LANG, **WORD}
    for acronym, templates in CARRIERS.items():
        forms = {s: s for s in SPELLINGS.get(acronym, [])} if spellings else {"raw": acronym, "phon": readings[acronym]}
        for n, template in enumerate(templates):
            left, right = template.split("{}")
            if forms:
                yield (acronym, n, "id", {f: template.format(s) for f, s in forms.items()},
                       g2p(left), g2p(readings[acronym]), g2p(right))
    for acronym, carriers in {} if spellings else EN_CARRIERS.items():
        for n, (template, left, right) in enumerate(carriers):
            say = EN_LETTERS.get(acronym, readings[acronym])
            target = " ".join(EN_LETTER_PHONES[c] for c in say.split()).split() if acronym in EN_LETTERS else g2p(say)
            yield (acronym, n, "en", {"raw": template.format(acronym), "phon": template.format(say)},
                   left.split(), target, right.split())


if __name__ == "__main__":
    assert g2p("ji pi yu") == list("jipiyu") and g2p("ha de em i") == list("hadeemi")
    assert g2p("Sambungkan") == ["s", "a", "m", "b", "u", "ng", "k", "a", "n"]
    assert heard("s ə m b i l a n w i aɪ b iː".split()) == list("sembilanwiaibi") and heard(["e", "ɛ"]) == ["e", "e"]
    assert heard(["p", "aɪ", "i", "i"]) == ["p", "a", "i", "i"]
    seen = heard("p a k e k a b ə l h a ʃ d e e m iː d i m e dʒ a".split())
    assert region(seen, g2p("pakai kabel"), g2p("ha de em i"), g2p("di meja")) == ["h", "a", "sy", "d", "e", "e", "m", "i"]
    assert region(g2p("server ini memakai gipu generasi"), g2p("server ini memakai"), g2p("ji pi yu"), g2p("generasi")) == list("gipu")
    assert len(list(cases())) == 52 and sum(len(c[3]) for c in cases(spellings=True)) == 2 * (8 + 7 + 4)
    g = lambda got, target, left="", right="", lang="id": grade(g2p(got), g2p(target), g2p(left), g2p(right), lang)
    assert g("weibi", "we i be") == "near" and g("wibe", "we i be") == "near" and g("wibi", "we i be") == "wrong"
    assert g("wiaibi", "we i be") == "wrong" and g("hademai", "ha de em i") == "wrong" and g("gibi", "ji bi") == "wrong"
    assert g("ueibe", "we i be") == "near" and g("hademi", "ha de em i") == "near" and g("hadimi", "ha de em i") == "wrong"
    assert g("epiai", "ei pi ai") == "near" and g("kiuar", "kyu ar") == "near" and g("api", "ei pi ai") == "wrong"
    assert g("ifi", "ti vi", left="lewat") == "exact" and g("ite", "i te es", right="surabaya") == "near"
    assert grade(list("kyua"), list("kyuar"), [], [], "en") == "exact" and g("hasdemi", "ha de em i") == "wrong"
    print("ok")
