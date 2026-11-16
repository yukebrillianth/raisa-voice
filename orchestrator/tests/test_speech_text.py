import asyncio

import pytest

from speech_text import normalize, screen_text

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
    ("Selamat datang di ITS", "Selamat datang di i te es"),
    ("Pameran UMKM", "Pameran u em ka em"),
    ("Saya RAISA", "Saya RAISA"),
    ("Tim IRIS dan robot NABBIT di ITS", "Tim IRIS dan robot NABBIT di i te es"),
    ("Fakultas FTEIC", "Fakultas ef te e i ce"),
    # emoji
    ("Halo 😊 selamat datang", "Halo  selamat datang"),
]


@pytest.mark.parametrize("written,spoken", CASES)
def test_normalize(written, spoken):
    assert normalize(written) == spoken


def test_table_rows_are_not_spoken():
    text = "Jadwalnya:\n| Hari | Jam |\n|---|---|\n| Senin | 08.00 |\nSilakan datang."
    assert normalize(text) == "Jadwalnya:\nSilakan datang."


def test_screen_text_strips_markdown_but_keeps_digits():
    shown = asyncio.run(screen_text("**Harga** tiket *Rp150.000*\n| a | b |"))
    assert shown == "Harga tiket Rp150.000"
