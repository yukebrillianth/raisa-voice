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
    ("Buka pukul 14.30 WIB.", "Buka pukul empat belas tiga puluh we i be."),
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
    ("Selamat datang di ITS", "Selamat datang di ITS"),
    ("Pameran UMKM", "Pameran UMKM"),
    ("Saya RAISA", "Saya Raisa"),
    ("Tim IRIS dan robot NABBIT di ITS", "Tim Iris dan robot Nabbit di ITS"),
    ("Fakultas FTEIC", "Fakultas FTEIC"),
    # emoji
    ("Halo 😊 selamat datang", "Halo  selamat datang"),
    # acronyms go raw to the voice; fixed overrides are the same in both languages
    ("Bayar pakai QRIS ya", "Bayar pakai kyuris ya"),
    ("Kirim file PDF ke email saya", "Kirim file PDF ke email saya"),
    ("Colok USB di sini", "Colok USB di sini"),
    ("Tonton di ITS TV", "Tonton di ITS TV"),
    ("Ruang VIP, GPU, API, QR, NFC, ID, IT dan GB", "Ruang VIP, GPU, API, QR, NFC, ID, IT dan GB"),
    ("Colokkan flashdisk ke port USB", "Cholokkan flashdisk ke port USB"),
    ("Kabelnya colokkan ke sini, cocok kan?", "Kabelnya cholokkan ke sini, cocok kan?"),
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
    ("Kabel HDMI di meja", "Kabel HDMI di meja"),
    ("Daftar di BRIN, SIM, dan UNAIR", "Daftar di Brin, Sim, dan Unair"),
    ("Ketua HIMATEKTRO dari BANDUNG", "Ketua Himatektro dari Bandung"),
    ("Kak ANDI dan DEWI ke MEDAN", "Kak Andi dan Dewi ke Medan"),
    ("Pukul 10.00 WITA dan 11.00 WIT", "Pukul sepuluh Wita dan sebelas we i te"),
    ("Mulai pukul 09.00 WIB.", "Mulai pukul sembilan we i be."),
    ("Bayar BPJS dan NPWP", "Bayar BPJS dan NPWP"),
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
