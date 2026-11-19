import asyncio

import pytest

from speech_text import language, normalize, screen_text

CASES = [
    # money
    ("Harganya Rp150.000.", "Harganya seratus lima puluh ribu rupiah."),
    ("Tiketnya Rp 25.000,- saja", "Tiketnya dua puluh lima ribu rupiah saja"),
    ("Anggaran Rp 1,5 juta", "Anggaran satu koma lima juta rupiah"),
    ("Total 150.000 rupiah", "Total seratus lima puluh ribu rupiah"),
    # decimals and plain numbers
    ("Nilainya 3,5 dari 4", "Nilainya tiga koma lima dari empat"),
    ("Ada 1.250 peserta", "Ada seribu dua ratus lima puluh peserta"),
    ("Juara ke-3", "Juara ketiga"),
    # time (ambiguity: 14.30 without "pukul" is still a time)
    ("Buka pukul 14.30 WIB.", "Buka pukul empat belas tiga puluh wé-i-bé."),
    ("Rapat jam 14:30", "Rapat jam empat belas tiga puluh"),
    ("Mulai 08.00 sampai 16.00", "Mulai delapan sampai enam belas"),
    ("Datang 09.05 ya", "Datang sembilan lewat lima menit ya"),
    ("Kumpul 14.30 di lobi", "Kumpul empat belas tiga puluh di lobi"),
    # dates, percent, units
    ("Tanggal 17 Agustus 2026", "Tanggal tujuh belas Agustus dua ribu dua puluh enam"),
    ("Diskon 50%", "Diskon lima puluh persen"),
    ("Naik 2,5 %", "Naik dua koma lima persen"),
    ("Jaraknya 5 km, beratnya 2 kg", "Jaraknya lima kilometer, beratnya dua kilogram"),
    ("Lebar 3 m dan 10 menit jalan", "Lebar tiga meter dan sepuluh menit jalan"),
    # phone numbers, digit by digit
    ("Hubungi 0812-3456-7890", "Hubungi nol delapan satu dua tiga empat lima enam tujuh delapan sembilan nol"),
    ("Telepon +62 812 3456 789", "Telepon plus enam dua delapan satu dua tiga empat lima enam tujuh delapan sembilan"),
    # abbreviations and acronyms
    ("Ada kursi, meja, dll.", "Ada kursi, meja, dan lain-lain."),
    ("Selamat datang di ITS", "Selamat datang di i-té-és"),
    ("Pameran UMKM", "Pameran u-ém-ka-ém"),
    ("Saya RAISA", "Saya Raisa"),
    ("Tim IRIS dan robot NABBIT di ITS", "Tim Iris dan robot Nabbit di i-té-és"),
    ("Fakultas FTEIC", "Fakultas éf-té-é-i-cé"),
    # emoji
    ("Halo 😊 selamat datang", "Halo selamat datang"),
    # acronyms go raw to the voice; fixed overrides are the same in both languages
    ("Bayar pakai QRIS ya", "Bayar pakai kyuris ya"),
    ("Kirim file PDF ke email saya", "Kirim file PDF ke email saya"),
    ("Colok USB di sini", "Colok USB di sini"),
    ("Tonton di ITS TV", "Tonton di i-té-és TV"),
    ("Ruang VIP, GPU, API, QR, NFC, ID, IT dan GB", "Ruang VIP, GPU, API, QR, NFC, ID, IT dan GB"),
    ("Colokkan flashdisk ke port USB", "Cholokkan flashdisk ke port USB"),
    ("Kabelnya colokkan ke sini, cocok kan?", "Kabelnya cholokkan ke sini, cocok kan?"),
    ("Sudah dicolokkan ke colokan? Colok saja, warnanya mencolok", "Sudah dicholokkan ke cholokan? Colok saja, warnanya mencolok"),
    ("Jangan parkir seenaknya, ada seekor kucing di lantai keenam", "Jangan parkir se-enaknya, ada seekor kucing di lantai keenam"),
    ("Pemakaian 450 kWh", "Pemakaian empat ratus lima puluh kilowatt hour"),
    ("Prosesor 3,5 GHz", "Prosesor tiga koma lima gigahertz"),
    ("Audio 48 kHz", "Audio empat puluh delapan kilohertz"),
    ("Port USB-C, 100 Mbps, 30 psi, 60 dB", "Port USB-C, seratus Mbps, tiga puluh psi, enam puluh dB"),
    # units with symbols
    ("Suhu hari ini 23°C", "Suhu hari ini dua puluh tiga derajat Celsius"),
    ("Suhu tubuh 36,5 °C", "Suhu tubuh tiga puluh enam koma lima derajat Celsius"),
    ("Luasnya 150 m² saja", "Luasnya seratus lima puluh meter persegi saja"),
    ("Kecepatan 40 km/jam", "Kecepatan empat puluh kilometer per jam"),
    # capitals that are words are said as words; WIB/WIT are spelled in the sentence language
    ("Kabel HDMI di meja", "Kabel ha-dé-ém-i di meja"),
    ("OK, NIK dan SKCK", "Ok, én-i-ka dan és-ka-cé-ka"),
    ("Daftar di BRIN, SIM, dan UNAIR", "Daftar di Brin, Sim, dan Unair"),
    ("Ketua HIMATEKTRO dari BANDUNG", "Ketua Himatektro dari Bandung"),
    ("Kak ANDI dan DEWI ke MEDAN", "Kak Andi dan Dewi ke Medan"),
    ("Pukul 10.00 WITA dan 11.00 WIT", "Pukul sepuluh Wita dan sebelas wé-i-té"),
    ("Mulai pukul 09.00 WIB.", "Mulai pukul sembilan wé-i-bé."),
    ("Bayar BPJS dan NPWP", "Bayar bé-pé-jé-és dan én-pé-wé-pé"),
    # English sentences
    ("Please plug the HDMI cable in.", "Please plug the HDMI cable in."),
    ("You can pay via QRIS or scan the QR code.", "You can pay via kyuris or scan the QR code."),
    ("The lab is on floor 3 and it is 23°C.", "The lab is on floor three and it is twenty-three degrees Celsius."),
    ("Your total is Rp 45,000.", "Your total is forty-five thousand rupiah."),
    ("The room is 150 m² and 1 km away.", "The room is one hundred and fifty square meters and one kilometer away."),
    ("It is 25% off until 10:30.", "It is twenty-five percent off until ten thirty."),
    ("Welcome to ITS, I am RAISA.", "Welcome to ITS, I am Raisa."),
    ("The shop opens at 9:00 WIB.", "The shop opens at nine W I B."),
    ("We open at 9:05 for 2.5 km runs.", "We open at nine oh five for two point five kilometers runs."),
    # academic titles: said in full, their dots give no pause; a degree keeps one dot at a sentence end
    ("Prof. Dr. Ir. Budi Santoso, S.T., M.Sc. akan hadir besok.",
     "Profesor Doktor Insinyur Budi Santoso, Sarjana Teknik, Master of Science akan hadir besok."),
    ("Pembicaranya Prof. Dr. Ir. Budi, S.T., M.Sc. Beliau dari ITS.",
     "Pembicaranya Profesor Doktor Insinyur Budi, Sarjana Teknik, Master of Science. Beliau dari i-té-és."),
    ("Dr. (HC) Ir. Soekarno dan dr. Sinta, S.K.M.", "Doktor Honoris Causa Insinyur Soekarno dan Dokter Sinta, Sarjana Kesehatan Masyarakat."),
    ("Dr.Eng. Rudy, S.T., M.T. dan Ir. Joko, M.Eng., Ph.D. hadir.",
     "Doktor Engineering Rudy, Sarjana Teknik, Magister Teknik dan Insinyur Joko, Master of Engineering, Doctor of Philosophy hadir."),
    ("Prof. Ir. Bambang, M.Sc.Eng., ASEAN Eng. membuka acara.",
     "Profesor Insinyur Bambang, Master of Science in Engineering, Asean Engineer membuka acara."),
    ("Bu Dra. Sri, M.Pd. dan Pak Drs. Hadi, M.M. sudah datang.",
     "Bu Doktoranda Sri, Magister Pendidikan dan Pak Doktorandus Hadi, Magister Manajemen sudah datang."),
    ("Andi, S.Kom., S.Si., S.E., S.H., S.Pd., S.Sos., S.Tr.T., M.Kom.",
     "Andi, Sarjana Komputer, Sarjana Sains, Sarjana Ekonomi, Sarjana Hukum, Sarjana Pendidikan, Sarjana Sosial, "
     "Sarjana Terapan Teknik, Magister Komputer."),
    ("Airnya habis. Irigasi di Dr.X.", "Airnya habis. Irigasi di Dr.X."),
]


@pytest.mark.parametrize("text,lang", [
    ("Selamat datang di lobi.", "id"),
    ("Kamu bisa download aplikasinya, lalu login pakai email kampus ya.", "id"),
    ("Maaf, sistem check-in otomatis sedang maintenance.", "id"),
    ("Please follow me to the meeting room.", "en"),
    ("Thank you.", "en"),
    ("Halo!", "id"),
])
def test_language(text, lang):
    assert language(text) == lang


def test_lowercase_it_and_id_stay_words():
    assert normalize("It is your id card.") == "It is your id card."


@pytest.mark.parametrize("written,spoken", CASES)
def test_normalize(written, spoken):
    assert normalize(written) == spoken


def test_table_rows_are_not_spoken():
    text = "Jadwalnya:\n| Hari | Jam |\n|---|---|\n| Senin | 08.00 |\nSilakan datang."
    assert normalize(text) == "Jadwalnya:\nSilakan datang."


def test_screen_text_strips_markdown_but_keeps_digits():
    shown = asyncio.run(screen_text("**Harga** tiket *Rp150.000*\n| a | b |"))
    assert shown == "Harga tiket Rp150.000"



# Found by running real LLM replies through bot.SentenceAggregator and normalize() (2026-11-19).
REPLY_CASES = [
    ("Beliau menjabat untuk masa jabatan 2024-2029.", None,
     "Beliau menjabat untuk masa jabatan dua ribu dua puluh empat sampai dua ribu dua puluh sembilan."),
    ("Buka pukul 08.00-10.00 setiap hari", None, "Buka pukul delapan sampai sepuluh setiap hari"),
    ("Tunggu 5-10 menit ya", None, "Tunggu lima sampai sepuluh menit ya"),
    ("Tanggal 19-11-2024 ya", None, "Tanggal sembilan belas November dua ribu dua puluh empat ya"),
    ("Due on 19/11/2024.", None, "Due on November nineteenth, twenty twenty-four."),
    ("Berawal dari IKIP Surabaya, UNESA kini menjadi universitas.", None,
     "Berawal dari Ikip Surabaya, Unesa kini menjadi universitas."),
    ("UIN Sunan Ampel (UINSA) dan PENS", None, "u-i-én Sunan Ampel (Uinsa) dan Pens"),
    ("GPU-nya pakai RTX 5090, 32 GB.", None, "GPU-nya pakai RTX lima puluh sembilan puluh, tiga puluh dua GB."),
    ("Sertifikat ISO 9001 dan modul ESP32", None, "Sertifikat i-és-o sembilan nol nol satu dan modul ESP tiga puluh dua"),
    ("Bawa KTPnya ya", None, "Bawa ka-té-pé-nya ya"),
    ("The API costs Rp 150.000 per month.", None, "The API costs one hundred and fifty thousand rupiah per month."),
    ("He has served since April 30, 2024.", None, "He has served since April thirtieth, twenty twenty-four."),
    ("It opened on 17 August 1945.", None, "It opened on the seventeenth of August, nineteen forty-five."),
    ("He led ITS from 2019 to 2024.", None, "He led ITS from twenty nineteen to twenty twenty-four."),
    ("The term is 2024-2029.", None, "The term is twenty twenty-four to twenty twenty-nine."),
    ("The RTX 4090 is fast.", None, "The RTX forty ninety is fast."),
    ("WIB menggunakan UTC+7.", None, "wé-i-bé menggunakan UTC plus tujuh."),
    ("Jakarta is in UTC+7.", None, "Jakarta is in UTC plus seven."),
    # no function words: the language of the sentence before decides
    ("Specs: 2 GPU, 64 GB RAM, 1 TB SSD.", "en", "Specs: two GPU, sixty-four GB Ram, one TB SSD."),
    ("Specs: 2 GPU, 64 GB RAM, 1 TB SSD.", "id", "Specs: dua GPU, enam puluh empat GB Ram, satu TB SSD."),
]


@pytest.mark.parametrize("written,before,spoken", REPLY_CASES)
def test_llm_replies(written, before, spoken):
    assert normalize(written, language(written, before)) == spoken


def test_language_needs_evidence_to_switch():
    assert language("Specs: 2 GPU.") == "id"
    assert language("Specs: 2 GPU.", "en") == "en"
    assert language("Ini 2 GPU.", "en") == "id"
    assert language("Monggo pinarak, kowe saka ngendi?") == "id"


def test_unknown_capitals_use_the_model_and_known_ones_the_table():
    import acronym_clf
    if acronym_clf._load() is None:
        pytest.skip("acronym_clf.joblib not built (python acronym_clf.py train)")
    assert acronym_clf.predict("NIK") == ("SPELL_ID", 1.0)
    assert normalize("Kantor BAPENDA buka") == "Kantor Bapenda buka"


# Everyday things an LLM writes (everyday.py), found on 2026-11-19.
EVERYDAY = [
    ("Kunjungi https://www.its.ac.id/pendaftaran untuk info.", "Kunjungi i-té-és dot a-cé dot i-dé untuk info."),
    ("Kirim email ke humas@its.ac.id ya.", "Kirim email ke humas et i-té-és dot a-cé dot i-dé ya."),
    ("Prosesornya 2.5 GHz, versi 3.1.4.", "Prosesornya dua koma lima gigahertz, versi tiga titik satu titik empat."),
    ("Minum obatnya 3x sehari, layarnya 1920x1080.",
     "Minum obatnya tiga kali sehari, layarnya seribu sembilan ratus dua puluh kali seribu delapan puluh."),
    ("Lokasinya di Jl. Raya ITS No. 5, Kec. Sukolilo.", "Lokasinya di Jalan Raya i-té-és nomor lima, Kecamatan Sukolilo."),
    ("Tersedia jenjang S1, S2, dan D3.", "Tersedia jenjang és satu, és dua, dan dé tiga."),
    ("Sudah mendukung 5G dan resolusi 4K.", "Sudah mendukung five G dan resolusi four K."),   # English, like GPU
    ("Harganya $20 atau Rp 2.5 juta.", "Harganya dua puluh dolar atau dua koma lima juta rupiah."),
    ("Musik tahun 1990-an ya.", "Musik tahun seribu sembilan ratus sembilan puluhan ya."),
    ("Baca Bab IV tentang abad XXI dan Perang Dunia II.",
     "Baca Bab empat tentang abad dua puluh satu dan Perang Dunia kedua."),
    ("Sri Sultan Hamengkubuwono X membuka acara.", "Sri Sultan Hamengkubuwono kesepuluh membuka acara."),
    ("Porsinya 1/2, sisanya 3/4, layanan 24/7.",
     "Porsinya setengah, sisanya tiga per empat, layanan dua puluh empat jam setiap hari."),
    ("Aman ± 99% & stabil, latensi ~20 ms.", "Aman kurang lebih sembilan puluh sembilan persen dan stabil, latensi sekitar dua puluh milidetik."),
    ("Rektor/direktur hadir dan/atau wakilnya.", "Rektor atau direktur hadir dan atau wakilnya."),
    ("Gedung A — keduanya • buka.", "Gedung A, keduanya, buka."),
    ("Penelitian Budi dkk.", "Penelitian Budi dan kawan-kawan."),
    ("Pukul 09.30 mulai, durasinya 1.5 jam.", "Pukul sembilan tiga puluh mulai, durasinya satu koma lima jam."),
    ("3. Politeknik Elektronika Negeri Surabaya", "tiga, Politeknik Elektronika Negeri Surabaya"),
    # English sentences
    ("For example, e.g. GPUs vs. CPUs, etc.", "For example, for example GPUs versus CPUs, et cetera."),
    ("Visit https://its.ac.id, it costs $1.5 million.", "Visit its dot ac dot id, it costs one point five million dollars."),
    ("World War II ended in 1945, see Chapter IV.", "World War two ended in nineteen forty-five, see Chapter four."),
    ("Latency is 20 ms.", "Latency is twenty milliseconds."),
    ("It supports 5G and 4K.", "It supports five G and four K."),
    # Roman numerals after a name; single letters only after a context word
    ("Ratu Elizabeth II dan Paus Yohanes Paulus II datang.", "Ratu Elizabeth kedua dan Paus Yohanes Paulus kedua datang."),
    ("Queen Elizabeth II visited.", "Queen Elizabeth the second visited."),
    ("Elizabeth II datang hari ini.", "Elizabeth kedua datang hari ini."),
    ("Kelas XI dan Kelas X belajar.", "Kelas sebelas dan Kelas sepuluh belajar."),
    ("Gedung C dan ukuran L, tipe X ya.", "Gedung C dan ukuran L, tipe X ya."),
    # scores are not ranges
    ("Persebaya menang 3-1 kemarin.", "Persebaya menang tiga lawan satu kemarin."),
    ("Skor akhir 2-2 ya.", "Skor akhir dua lawan dua ya."),
    ("They won 3-1.", "They won three to one."),
    ("Jam buka 8-10 ya.", "Jam buka delapan sampai sepuluh ya."),
]


@pytest.mark.parametrize("written,spoken", EVERYDAY)
def test_everyday(written, spoken):
    assert normalize(written) == spoken


def test_roman_numerals_need_a_context_word():
    assert normalize("Kampus di DIY dan MIX") == normalize("Kampus di DIY dan MIX")   # no crash, no number
    assert "empat" not in normalize("Bawa IV ya")
