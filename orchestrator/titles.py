"""Titles around a name, said in full: "Prof. Dr. (H.C.) Ir. Bambang, S.T., Ph.D." -> spoken text.

Rule-based on purpose. A title has exactly one right reading, so the reading comes from a lexicon
(title_lexicon.json, built and evaluated by train/titles/build_title_lexicon.py) plus a small grammar,
never from a model that can invent an expansion.

Grammar
- Before a name: a chain of pre titles (Prof., Dr., (H.C.), Ir., dr., H., Mr. ...) followed by a
  capitalised name. "H." is also a common initial, so it is not a title right after a capitalised
  word ("Bambang H. Santoso"), except after a form of address ("Bapak H. Agus").
- After a name: ", title" repeated. The first one must follow a name in Title Case ("Budi, S.T."),
  so "ITS, UGM" or "seri ST, MT" are left alone. A compact dotted title right after a name without a
  comma ("Budi S.T. hadir") also counts, unless a capitalised word follows ("S.H. Wibowo" = initials).
- Indonesian degrees are composed: level (S., S.Tr., M., M.Tr., A.Md., A.Ma., A.P., Sp.) + field
  code (Kom, T, KM, I.Kom ...). Explicit entries (Ph.D., M.Sc.Eng., IPU, ASEAN Eng.) win over
  composed ones. "Sp.PD-KGEH" and "Sp.M(K)" add the consultant part.
- Writing is forgiving: "S.T.", "S. T.", "ST", "Ph.D", "PhD" are one title.
- Unknown dotted degrees inside a chain (or starting with a known level: "S.Xyz.") are spelled out
  and logged to TITLE_LOG_PATH, so the lexicon can grow from what the LLM actually writes.

Language: "id" (Indonesian, Javanese and code-switched sentences) and "en". Code-switched sentences
use Indonesian names for Indonesian degrees and keep English degrees English (natural Indonesian
speech). English sentences use English names: "S.T." -> "Bachelor of Engineering", "Prof." -> "Professor".

Post-name mode (TITLE_POST_MODE, or the `mode` argument): expand (default) | short | drop.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase

log = logging.getLogger(__name__)

LEXICON_PATH = Path(os.environ.get("TITLE_LEXICON_PATH", Path(__file__).resolve().parent / "title_lexicon.json"))
POST_MODE = os.environ.get("TITLE_POST_MODE", "expand")
TITLE_LOG_PATH = os.environ.get("TITLE_LOG_PATH", "")

# Letter names for short readings in Indonesian sentences, joined by "-" like the acronym rule.
ID_LETTERS = dict(zip(ascii_uppercase, "a bé cé dé é éf gé ha i jé ka él ém én o pé ki ér és té u vé wé éks yé zét".split()))
EN_LETTERS = dict(zip(ascii_uppercase, "ei bi si di i ef ji eich ai jei kei el em en ou pi kyu ar es ti yu vi dabelyu eks wai zi".split()))

SEP = r"(?:\.[ \t]?-?|[ \t]|-)?"      # between parts of one title: "S.T", "S. T", "ST", "THT-KL", "Dr.-Ing"
END = r"(?:\.|(?![A-Za-z0-9]))"         # a title ends with its dot or at a word boundary
VOWEL = re.compile(r"[aeiouAEIOU]")
WORDS = re.compile(r"[A-Za-z]+")
COMMA = re.compile(r",[ \t]*")
GAP = re.compile(r"(?<=[a-zà-ÿ])[ \t]+(?=[A-Z])")
NAME_BEFORE = re.compile(r"(?:^|[\s(\"'“])[A-Z][a-zà-ÿ][\w'’\-]*$")
WORD_BEFORE = re.compile(r"([A-Za-zÀ-ÿ]+)[ \t]+$")
NAME_AFTER = re.compile(r"[ \t]+(?=[A-Z])")
SPACES = re.compile(r"[ \t]*")
HC = re.compile(r"\([ \t]?[Hh]\.?[ \t]?[Cc]\.?[ \t]?\)|[Hh]\.[ \t]?[Cc]\.|HC(?![A-Za-z0-9])")
# Said where they stand, e.g. "Kol. TNI (Purn.) Ir. Marseno": the dot must not end the sentence either.
PARENS = {"Purn": ("Purnawirawan", "Retired"), "Ret": ("Purnawirawan", "Retired")}
PAREN = re.compile(r"\([ \t]?(" + "|".join(PARENS) + r")\.?[ \t]?\)")
KONSULTAN_K = re.compile(r"[ \t]?\([ \t]?K[ \t]?\)")
UNKNOWN = re.compile(r"(?:[A-Z][A-Za-z]{0,5}\.){1,4}(?:[A-Z][A-Za-z]{0,5}(?![A-Za-z0-9]))?")
# Forms of address after which "H." is Haji, not an initial.
ADDRESS = {"Bapak", "Pak", "Ibu", "Bu", "Saudara", "Saudari", "Kak", "Mas", "Mbak", "Terhormat", "Almarhum",
           "Almarhumah", "Alm", "Almh", "Beliau", "Tuan", "Nyonya", "Yth"}


def key(text: str) -> str:
    """Letters only: every way of writing one title maps to one key ("S. Kom." -> "SKom")."""
    return "".join(WORDS.findall(text))


def parts(surface: str) -> list[str]:
    """Matching units: capitals up to 4 letters split into letters, so "KM" also matches "K.M."."""
    out: list[str] = []
    for w in WORDS.findall(surface):
        out.extend(w if w.isupper() and len(w) <= 4 else [w])
    return out


def short_form(surface: str, letters: str, lang: str) -> str:
    """'S.Kom.' -> 'és-kom', 'Ph.D.' -> 'pi-eich-di'; in English sentences 'S Kom', 'P H D'."""
    out: list[str] = []
    for w in WORDS.findall(surface):
        if len(w) > 1 and VOWEL.search(w) and not (w.isupper() and len(w) <= 4):
            out.append(w.lower())
        elif lang == "en":
            out.extend(w.upper())
        else:
            out.extend((EN_LETTERS if letters == "en" else ID_LETTERS)[c] for c in w.upper())
    return (" " if lang == "en" else "-").join(out)


@dataclass(frozen=True)
class Title:
    surface: str
    kind: str                      # pre | post | spesialis | konsultan
    id: str | None
    en: str | None
    short_id: str
    short_en: str
    explicit: bool = True
    initial: bool = False
    dot_required: bool = False
    chain_only_dotless: bool = False
    chain_only: bool = False

    def say(self, lang: str, mode: str = "expand") -> str:
        if mode == "short" and self.kind != "pre":
            return self.short_en if lang == "en" else self.short_id
        full = self.en if lang == "en" else self.id
        return full or self.id or (self.short_en if lang == "en" else self.short_id)


def trie_pattern(sequences: list[list[str]]) -> str:
    """One regex for many titles, factored by shared prefixes; longer titles are tried first."""
    root: dict = {}
    for seq in sequences:
        node = root
        for p in seq:
            node = node.setdefault(p, {})
        node[""] = {}

    def emit(node: dict, first: bool) -> str:
        alts = [("" if first else SEP) + re.escape(p) + emit(child, False)
                for p, child in sorted(((p, c) for p, c in node.items() if p), key=lambda pc: (-len(pc[0]), pc[0]))]
        if not alts:
            return ""
        body = "(?:" + "|".join(alts) + ")"
        return body + "?" if "" in node else body

    return emit(root, True)


EN_HINTS = {"Sc", "Eng", "Ph", "App", "Tech", "Ed", "Phil", "Arch", "Agr", "Com", "Bus", "Admin"}


@dataclass(frozen=True)
class _Spelled:
    """A known title followed by parts nobody listed: say the first, spell the rest."""
    known: Title
    rest: str
    letters: str
    kind: str = "post"

    def say(self, lang: str, mode: str = "expand") -> str:
        return f"{self.known.say(lang, mode)} {short_form(self.rest, self.letters, lang)}"


@dataclass
class _Hit:
    start: int
    end: int
    raw: str
    title: Title | _Spelled | None
    extra: Title | None = None     # consultant part of a specialist
    consultant: bool = False       # "(K)"
    letters: str = "id"            # alphabet for spelling an unknown degree

    def say(self, lang: str, mode: str) -> str:
        if self.title is None:     # unknown degree: spelled out
            return short_form(self.raw, self.letters, lang)
        if mode == "short":
            return short_form(self.raw, "id", lang) if (self.extra or self.consultant) else self.title.say(lang, mode)
        text = self.title.say(lang, mode)
        if self.extra:
            text += (f", Consultant in {self.extra.say(lang)}" if lang == "en" else f" Konsultan {self.extra.say(lang)}")
        elif self.consultant:
            text += ", Consultant" if lang == "en" else " Konsultan"
        return text


class TitleReader:
    def __init__(self, data: dict):
        self.pre: dict[str, Title] = {}
        self.post: dict[str, Title] = {}
        self.konsultan: dict[str, Title] = {}
        self.collisions: list[tuple[str, str, str]] = []
        tables = data.get("tables", {})
        self.level_keys = {key(lv["code"]) for lv in data.get("levels", [])}

        for e in data.get("post", []):
            self._add(self.post, self._entry(e, "post"))
        for e in data.get("pre", []):
            self._add(self.pre, self._entry(e, "pre"))
        for e in tables.get("konsultan", []):
            self._add(self.konsultan, self._entry({**e, "surface": e["code"]}, "konsultan", explicit=False))
        for lv in data.get("levels", []):
            table = lv.get("table", "field")
            for f in tables.get(table, []):
                surface = f"{lv['code']}.{f['code']}."
                id_ = f"{lv['id']} {f['id']}"
                en = f"{lv['en']} {f['en']}" if lv.get("en") and f.get("en") else id_
                kind = "spesialis" if table == "spesialis" else "post"
                self._add(self.post, Title(surface, kind, id_, en, short_form(surface, "id", "id"),
                                           short_form(surface, "id", "en"), explicit=False))

        pre_trie = trie_pattern([parts(t.surface) for t in self.pre.values()])
        post_trie = trie_pattern([parts(t.surface) for t in self.post.values()])
        kons_trie = trie_pattern([parts(t.surface) for t in self.konsultan.values()])
        self.pre_at = re.compile(rf"(?:{pre_trie}){END}")
        self.pre_find = re.compile(rf"(?<![\w.])(?=(?:{pre_trie}){END})")
        self.post_at = re.compile(rf"(?:{post_trie}){END}")
        self.kons_at = re.compile(rf"-[ \t]?K[ \t]?(?:{kons_trie}){END}") if self.konsultan else None
        # Dots that never end a sentence: a title's, "(H.C.)", and one glued to the next letter.
        self.dots = re.compile(rf"(?<![\w.])(?:{pre_trie}|{post_trie}){END}|{HC.pattern}|{PAREN.pattern}|\.(?=[A-Za-z])")

    @staticmethod
    def _entry(e: dict, kind: str, explicit: bool = True) -> Title:
        surface, letters = e["surface"], e.get("letters", "id")
        return Title(
            surface=surface, kind=kind, id=e.get("id"), en=e.get("en") or e.get("id"),
            short_id=e.get("short") or short_form(surface, letters, "id"),
            short_en=e.get("short_en") or short_form(surface, letters, "en"),
            explicit=explicit, initial=e.get("initial", False), dot_required=e.get("dot_required", False),
            chain_only_dotless=e.get("chain_only_dotless", False), chain_only=e.get("chain_only", False),
        )

    def _add(self, store: dict[str, Title], t: Title) -> None:
        k = key(t.surface)
        old = store.get(k)
        if old is None or (t.explicit and not old.explicit):
            store[k] = t
        elif (old.id, old.en) != (t.id, t.en) and old.explicit == t.explicit:
            self.collisions.append((k, old.surface, t.surface))

    # ------------------------------------------------------------------ before the name

    def _address_ok(self, text: str, start: int) -> bool:
        """'H.' is Haji at a sentence start, after a lowercase word or after 'Bapak'.
        After a capitalised word ('Bambang H. Santoso') or an abbreviation ('S. H. Wibowo') it is an initial."""
        m = re.search(r"(\S+)[ \t]+$", text[:start])
        if not m:
            return True
        token = m.group(1)
        if token[-1] in ",;:!?(\"'“":
            return True
        word = token.rstrip(".")
        if token.endswith("."):
            return len(word) > 1 and word.isalpha() and word.islower()
        return not word[:1].isupper() or word in ADDRESS

    def _pre(self, text: str, lang: str, taken: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
        edits: list[tuple[int, int, str]] = []
        last = 0
        for m in self.pre_find.finditer(text):
            start = m.start()
            if start < last or any(a <= start < b for a, b in taken):
                continue
            chain: list[Title] = []
            pos = end = start
            while True:
                mm = self.pre_at.match(text, pos)
                t = self.pre.get(key(mm.group())) if mm else None
                if t and ((t.dot_required and not mm.group().endswith(".")) or (t.chain_only and not chain)):
                    t = None
                if t is None and chain:
                    mm = HC.match(text, pos)
                    t = Title("H.C.", "pre", "Honoris Causa", "Honoris Causa", "", "") if mm else None
                if t is None:
                    break
                chain.append(t)
                end = mm.end()
                gap = SPACES.match(text, end).end()
                if gap == end and text[end - 1] != ".":
                    break
                pos = gap
            if not chain or not NAME_AFTER.match(text, end):
                continue
            if chain[0].initial and not self._address_ok(text, start):
                continue
            edits.append((start, end, " ".join(t.say(lang) for t in chain)))
            last = end
        return edits

    # ------------------------------------------------------------------ after the name

    def _hit(self, text: str, pos: int, in_chain: bool, compact: bool = False) -> _Hit | None:
        m = self.post_at.match(text, pos)
        if m:
            raw = m.group()
            t = self.post.get(key(raw))
            if t and m.end() < len(text) and text[m.end()].isalnum():
                t = None
            if t and t.chain_only_dotless and not in_chain and "." not in raw[:-1]:
                t = None
            if t and compact and (" " in raw or "." not in raw or len(parts(raw)) < 2):
                t = None
            if t:
                hit = _Hit(pos, m.end(), raw, t)
                if t.kind == "spesialis":
                    k = self.kons_at.match(text, hit.end) if self.kons_at else None
                    kt = self.konsultan.get(key(k.group())[1:]) if k else None
                    if kt:
                        hit.extra, hit.end, hit.raw = kt, k.end(), text[pos:k.end()]
                    elif kk := KONSULTAN_K.match(text, hit.end):
                        hit.consultant, hit.end, hit.raw = True, kk.end(), text[pos:kk.end()]
                return hit
        if compact:
            return None
        u = UNKNOWN.match(text, pos)
        if u:
            words = WORDS.findall(u.group())
            known_level = any(key(".".join(words[:n])) in self.level_keys for n in (1, 2))
            if len(words) >= 2 and (in_chain or known_level) and not (u.end() < len(text) and text[u.end()].isalnum()):
                self._log_unknown(u.group())
                return self._partly_known(text, pos, u.end())
        return None

    def _partly_known(self, text: str, pos: int, end: int) -> _Hit:
        """'M.Sc.ES.': read the 'M.Sc.' we know, spell 'ES' in the same alphabet ('Master of Science i-es')."""
        raw = text[pos:end]
        words = WORDS.findall(raw)
        for n in range(len(words) - 1, 0, -1):
            t = self.post.get("".join(words[:n]))
            if t and t.explicit:
                rest = ".".join(words[n:]) + "."
                letters = "en" if t.short_id == short_form(t.surface, "en", "id") else "id"
                return _Hit(pos, end, raw, _Spelled(t, rest, letters))
        english = any(w in EN_HINTS for w in words)
        return _Hit(pos, end, raw, None, letters="en" if english else "id")

    def _post(self, text: str, lang: str, mode: str) -> list[tuple[int, int, str]]:
        anchors = sorted([(m.start(), m.end(), False) for m in COMMA.finditer(text)]
                         + [(m.start(), m.end(), True) for m in GAP.finditer(text)])
        edits: list[tuple[int, int, str]] = []
        done = 0
        for a_start, a_end, compact in anchors:
            if a_start < done or not NAME_BEFORE.search(text, 0, a_start):
                continue
            first = self._hit(text, a_end, in_chain=False, compact=compact)
            if not first:
                continue
            if compact and NAME_AFTER.match(text, first.end):   # "Bapak S.H. Wibowo": initials
                continue
            chain = [first]
            while (c := COMMA.match(text, chain[-1].end)) and (nxt := self._hit(text, c.end(), in_chain=True)):
                chain.append(nxt)
            last = chain[-1]
            dot = "." if self._sentence_end(text, last) else ""
            if mode == "drop":
                edits.append((a_start, last.end, dot))
            else:
                for h in chain:
                    edits.append((h.start, h.end, h.say(lang, mode) + (dot if h is last else "")))
            done = last.end
        return edits

    def _sentence_end(self, text: str, hit: _Hit) -> bool:
        """A title's last dot also ends the sentence when nothing, or a new capitalised sentence, follows."""
        if not hit.raw.endswith("."):
            return False
        rest = text[hit.end:]
        if not rest.strip() or rest[:1] == "\n":
            return True
        nxt = re.match(r"[ \t]+([A-Z])", rest)
        if not nxt:
            return False
        m = self.pre_at.match(text, hit.end + nxt.start(1))
        return not (m and key(m.group()) in self.pre)   # "..., S.T. Dr. Ani" is still one sentence

    def _log_unknown(self, raw: str) -> None:
        if TITLE_LOG_PATH:
            try:
                with open(TITLE_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(raw + "\n")
            except OSError:
                pass

    # ------------------------------------------------------------------ public

    def standalone(self, word: str, lang: str, mode: str) -> str | None:
        """One written title on its own (word_times passes single words)."""
        core = word.rstrip(",;:!?")
        for store, pattern in ((self.post, self.post_at), (self.pre, self.pre_at)):
            m = pattern.fullmatch(core)
            t = store.get(key(core)) if m else None
            if t:
                return t.say(lang, mode) + word[len(core):]
        return None

    def say(self, text: str, lang: str = "id", mode: str | None = None) -> str:
        mode = mode or POST_MODE
        if text and not any(c.isspace() for c in text.strip()):
            alone = self.standalone(text.strip(), lang, mode)
            if alone is not None:
                return text.replace(text.strip(), alone)
        # Both passes read the original text, so neither rereads what the other wrote.
        post = self._post(text, lang, mode)
        edits = post + self._pre(text, lang, [(a, b) for a, b, _ in post])
        edits += [(m.start(), m.end(), "(" + PARENS[m.group(1)][lang == "en"] + ")") for m in PAREN.finditer(text)]
        for start, end, new in sorted(edits, reverse=True):
            text = text[:start] + new + text[end:]
        return text


def load(path: Path = LEXICON_PATH) -> TitleReader:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    overlay = os.environ.get("SPEECH_LEXICON_PATH", "")
    if overlay and Path(overlay).exists():   # tenant titles: {"titles": {"post": [...], "pre": [...]}}
        try:
            extra = json.loads(Path(overlay).read_text(encoding="utf-8")).get("titles", {})
            for section in ("post", "pre", "levels"):
                data[section] = extra.get(section, []) + data.get(section, [])
            for name, rows in extra.get("tables", {}).items():
                data.setdefault("tables", {})[name] = rows + data["tables"].get(name, [])
        except Exception as exc:  # noqa: BLE001 - a bad overlay must not take the voice down
            log.warning("Ignoring titles in %s: %s", overlay, exc)
    return TitleReader(data)


READER = load()
TITLE_DOTS = READER.dots


def say_titles(text: str, lang: str = "id", mode: str | None = None) -> str:
    return READER.say(text, lang, mode)
