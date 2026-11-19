import pytest

from titles import READER, TITLE_DOTS, say_titles

BAMBANG = "Prof. Dr. (H.C.) Ir. Bambang Pramujati, S.T., M.Sc.Eng., Ph.D., IPU, ASEAN Eng."

CASES = [
    # the full chain, three modes, Indonesian and English
    (BAMBANG, "id", "expand",
     "Profesor Doktor Honoris Causa Insinyur Bambang Pramujati, Sarjana Teknik, Master of Science in Engineering, "
     "Doctor of Philosophy, Insinyur Profesional Utama, Asean Engineer."),
    (BAMBANG, "id", "short",
     "Profesor Doktor Honoris Causa Insinyur Bambang Pramujati, és-té, em-es-si-eng, pi-eich-di, i-pé-u, Asean Eng."),
    (BAMBANG, "id", "drop", "Profesor Doktor Honoris Causa Insinyur Bambang Pramujati."),
    ("Please welcome " + BAMBANG, "en", "expand",
     "Please welcome Professor Doctor Honoris Causa Engineer Bambang Pramujati, Bachelor of Engineering, "
     "Master of Science in Engineering, Doctor of Philosophy, Insinyur Profesional Utama, Asean Engineer."),
    # code-switched sentences are read as Indonesian; English degrees stay English
    ("Next speaker kita Dr. Rina, M.Kom., Ph.D., beliau expert AI.", "id", "expand",
     "Next speaker kita Doktor Rina, Magister Komputer, Doctor of Philosophy, beliau expert AI."),
    # sloppy writing
    ("Budi, ST, MT hadir.", "id", "expand", "Budi, Sarjana Teknik, Magister Teknik hadir."),
    ("Budi, S. Kom., M. Kom hadir.", "id", "expand", "Budi, Sarjana Komputer, Magister Komputer hadir."),
    ("Dr Budi, PhD hadir.", "id", "expand", "Doktor Budi, Doctor of Philosophy hadir."),
    ("Budi S.T. hadir.", "id", "expand", "Budi Sarjana Teknik hadir."),
    # the sentence-end dot
    ("Dosennya Ani, S.Kom. Beliau ramah.", "id", "expand", "Dosennya Ani, Sarjana Komputer. Beliau ramah."),
    ("Dosennya Ani, S.Kom. dari ITS.", "id", "expand", "Dosennya Ani, Sarjana Komputer dari ITS."),
    # medicine
    ("dr. Ani, Sp.PD-KGEH, FINASIM praktik.", "id", "expand",
     "Dokter Ani, Spesialis Penyakit Dalam Konsultan Gastroenterohepatologi, "
     "Fellow of the Indonesian Society of Internal Medicine praktik."),
    ("dr. Ani, Sp.M(K) praktik.", "id", "expand", "Dokter Ani, Spesialis Mata Konsultan praktik."),
    ("drg. Sinta, Sp.Ort. ada.", "id", "expand", "Dokter Gigi Sinta, Spesialis Ortodonsia ada."),
    # composed degrees nobody listed one by one
    ("Wati, A.Md.Kep., S.Tr.Kes. datang.", "id", "expand",
     "Wati, Ahli Madya Keperawatan, Sarjana Terapan Kesehatan datang."),
    ("Wati, A.Md.Kep. came.", "en", "expand", "Wati, Associate Degree in Nursing came."),
    # religious and forms of address
    ("Bapak H. Agus Salim hadir.", "id", "expand", "Bapak Haji Agus Salim hadir."),
    ("Ceramah oleh K.H. Hasyim.", "id", "expand", "Ceramah oleh Kiai Haji Hasyim."),
    # from real LLM replies (rectors of ITS): markdown bold leaves "Ph.D.." behind
    ("adalah Prof. Ir. Bambang Pramujati, S.T., M.Sc.Eng., Ph.D..", "id", "expand",
     "adalah Profesor Insinyur Bambang Pramujati, Sarjana Teknik, Master of Science in Engineering, Doctor of Philosophy."),
    ("Prof. Ir. R. Soemadijo (periode 1968-1973)", "id", "expand", "Profesor Insinyur Raden Soemadijo (periode 1968-1973)"),
    ("Kol. TNI (Purn.) Ir. Marseno Wirjosapoetra", "id", "expand", "Kolonel TNI (Purnawirawan) Insinyur Marseno Wirjosapoetra"),
    ("Prof. Ir. Joni Hermana, M.Sc.ES., Ph.D. (periode 2015-2019)", "id", "expand",
     "Profesor Insinyur Joni Hermana, Master of Science i-es, Doctor of Philosophy (periode 2015-2019)"),
    ("Prof. Dr. Ir. Triyogi Yuwono, DEA. (periode 2011-2015)", "id", "expand",
     "Profesor Doktor Insinyur Triyogi Yuwono, dé-é-a (periode 2011-2015)"),
    ("Prof. Ir. Priyo Suprobo, MS., Ph.D. (periode 2007-2011)", "id", "expand",
     "Profesor Insinyur Priyo Suprobo, Magister Sains, Doctor of Philosophy (periode 2007-2011)"),
    # from the Surabaya rectors reply (2026-11-19)
    ("Prof. Dr. Muhammad Madyan, S.E., M.Si., M.Fin. (periode 2025-2030)", "id", "expand",
     "Profesor Doktor Muhammad Madyan, Sarjana Ekonomi, Magister Sains, Master of Finance (periode 2025-2030)"),
    ("Dr. Sri Rahayu, S.Tr.Keb., S.Kep., Ners., M.Kes. (periode 2026-2030)", "id", "expand",
     "Doktor Sri Rahayu, Sarjana Terapan Kebidanan, Sarjana Keperawatan, Ners, Magister Kesehatan (periode 2026-2030)"),
    ("Lutfi Rusyadi, SKM, M.HKes, M.Sc. (periode 2022-2026)", "id", "expand",
     "Lutfi Rusyadi, Sarjana Kesehatan Masyarakat, Magister Hukum Kesehatan, Master of Science (periode 2022-2026)"),
    # unknown degree inside a chain: spelled, never invented
    ("Budi, M.Kom., M.Xyz. hadir.", "id", "expand", "Budi, Magister Komputer, ém-éks-yé-zét hadir."),
]

UNTOUCHED = [
    "Kampus ITS, UGM, dan UI sudah buka.",
    "Motor seri ST, MT, dan SE tersedia.",
    "Bambang H. Santoso hadir hari ini.",
    "Bapak S.H. Wibowo sudah datang.",
    "Saya ketemu S. H. Wibowo kemarin.",
    "Airnya habis. Irigasi di Dr.X.",
    "The office is in San Jose, CA.",
    "Baltimore, MD is in the US.",
    "Tim Budi, Ani, dan Sinta menang.",
    "Ukuran S, M, dan L tersedia.",
    "Bapak R. Soeprapto hadir.",          # R. alone is an initial, Raden only inside a chain
]


@pytest.mark.parametrize("written,lang,mode,spoken", CASES)
def test_titles(written, lang, mode, spoken):
    assert say_titles(written, lang, mode) == spoken


@pytest.mark.parametrize("written", UNTOUCHED)
def test_look_alikes_stay(written):
    lang = "en" if written.startswith(("The", "Baltimore")) else "id"
    assert say_titles(written, lang, "expand") == written


def test_one_word_at_a_time():
    """word_times() normalizes word by word."""
    assert say_titles("S.T.,") == "Sarjana Teknik,"
    assert say_titles("Prof.") == "Profesor"


def test_title_dots_cover_the_whole_chain():
    """bot.TitleDots keeps the splitter off every dot up to the end of each match."""
    protected = {m.end() for m in TITLE_DOTS.finditer(BAMBANG)}
    for title in ["Prof.", "Dr.", "(H.C.)", "Ir.", "S.T.", "M.Sc.Eng.", "Ph.D.", "ASEAN Eng."]:
        assert BAMBANG.index(title) + len(title) in protected, title


def test_unknown_degree_does_not_end_the_sentence():
    text = "Rektor: Budi, M.Xyz. (periode 2025-2030)"
    assert any(m.group().endswith("M.Xyz.") for m in TITLE_DOTS.finditer(text))


def test_lexicon_has_no_clashing_spellings():
    assert READER.collisions == []
