"""Writes the acronym training manifest for LoRA v3.

    python3 train/v3/acronym_manifest.py ../dataset-baru/voxcpm2_lora_v3_acronym.jsonl

Why: an audit of the 600 clips (out/acronym_audit_600.json) showed the training audio itself reads
GPU as "ge pe u", GB as "gigabyte", QRIS half the time letter by letter, and most acronyms occur
1-3 times. Each row here carries a pronunciation note for the generator (`hint`) and the reading a
judge must hear before the clip may be used (`expect`: acronym -> accepted spellings, spaces ignored).
"""

import json
import re
import sys

# acronym: (reading told to the generator, accepted judge spellings without spaces)
ID = {
    # always English letter names, also in Indonesian sentences (user's rule)
    "USB": ("yu es bi", ["yuesbi"]), "TV": ("ti vi", ["tivi"]), "GPU": ("ji pi yu", ["jipiyu"]),
    "GB": ("ji bi", ["jibi"]), "API": ("ei pi ai", ["eipiai"]), "QR": ("kyu ar", ["kyuar", "kiuar"]),
    "NFC": ("en ef si", ["enefsi"]), "VIP": ("vi ai pi", ["viaipi"]), "ID": ("ai di", ["aidi"]),
    "IT": ("ai ti", ["aiti"]), "CPU": ("si pi yu", ["sipiyu"]), "SSD": ("es es di", ["esesdi"]),
    "AI": ("ei ai", ["eiai"]), "LED": ("el i di", ["elidi"]), "PC": ("pi si", ["pisi"]),
    # fixed readings
    "QRIS": ("kyuris, satu kata", ["kyuris", "kiuris"]), "PDF": ("pi di ef", ["pidief"]),
    # Indonesian letter names
    "ITS": ("i te es", ["itees"]), "WIB": ("we i be", ["weibe"]), "WIT": ("we i te", ["weite"]),
    "HDMI": ("ha de em i", ["hadeemi"]), "HD": ("ha de", ["hade"]), "KTP": ("ka te pe", ["katepe"]),
    "KTM": ("ka te em", ["kateem"]), "UMKM": ("u em ka em", ["uemkaem"]), "BPJS": ("be pe je es", ["bepejees"]),
    "NPWP": ("en pe we pe", ["enpewepe"]), "BUMN": ("be u em en", ["beuemen"]), "PKM": ("pe ka em", ["pekaem"]),
    "IPK": ("i pe ka", ["ipeka"]), "KRS": ("ka er es", ["kaeres"]), "FTEIC": ("ef te e i ce", ["efteeice"]),
    "UGM": ("u ge em", ["ugeem"]), "ITB": ("i te be", ["itebe"]), "UI": ("u i", ["ui"]),
    "BNI": ("be en i", ["beeni"]), "BRI": ("be er i", ["beeri"]), "BCA": ("be ce a", ["becea"]),
    # read as one word
    "WITA": ("wita, satu kata", ["wita"]), "RAISA": ("raisa, satu kata", ["raisa"]),
    "NABBIT": ("nabit, satu kata", ["nabit", "nabbit"]), "BIMA": ("bima, satu kata", ["bima"]),
    "LUNA": ("luna, satu kata", ["luna"]), "UNAIR": ("unair, satu kata", ["unair"]),
    "BRIN": ("brin, satu kata", ["brin"]), "SIM": ("sim, satu kata", ["sim"]), "RAM": ("ram, satu kata", ["ram"]),
    "LAN": ("lan, satu kata", ["lan"]), "EXPO": ("ekspo, satu kata", ["ekspo"]),
    "SPARK": ("spark, satu kata", ["spark"]), "HIMATEKTRO": ("himatektro, satu kata", ["himatektro"]),
    "MAGE": ("ma ge, satu kata", ["mage"]), "IRIS": ("ai ris, satu kata", ["airis"]),
    "SKEM": ("skem, satu kata", ["skem"]),
}
# English sentences: same as above except these two follow the sentence's language
EN = {**ID, "HDMI": ("eich di em ai", ["eichdiemai", "eijdiemai", "eitchdiemai", "ecdiemai"]),
      "ITS": ("ai ti es", ["aities"])}

SENTENCES_ID = """
Kabel USB untuk mengisi daya ponsel ada di laci meja resepsionis.
Silakan colokkan flashdisk Anda ke port USB di samping layar.
USB yang tertinggal di ruang rapat sudah saya titipkan ke petugas keamanan.
Maaf, port USB di meja ini sedang rusak, silakan pakai yang di sebelah.
Proyektor di aula bisa membaca berkas langsung dari USB.
Kalau butuh pengisi daya, saya punya adaptor dengan dua lubang USB.
Jadwal acara hari ini ditampilkan di TV besar dekat pintu masuk.
TV di ruang tunggu sedang menayangkan profil kampus.
Silakan duduk dulu, Anda bisa menonton TV sambil menunggu giliran.
Remote untuk TV ruang rapat disimpan di rak bawah.
Siaran langsung wisuda bisa ditonton lewat TV di lobi.
Maaf, suara TV saya kecilkan dulu supaya kita bisa mengobrol.
Server riset kami memakai GPU terbaru untuk melatih model kecerdasan buatan.
GPU di laboratorium ini bekerja hampir tanpa henti sepanjang minggu.
Tanpa GPU, proses pengenalan suara saya akan jauh lebih lambat.
Mahasiswa bisa meminjam waktu pakai GPU lewat portal laboratorium.
Suhu GPU selalu dipantau agar server tidak kepanasan.
Otak saya berjalan di satu GPU yang ada di ruang server.
Memori server ini berkapasitas tiga puluh dua GB.
Kuota internet tamu dibatasi dua GB per hari.
Flashdisk hadiah untuk peserta berukuran enam belas GB.
Berkas video presentasinya cukup besar, sekitar empat GB.
Ponsel Anda masih punya ruang kosong delapan GB.
Paket data seratus GB itu berlaku selama satu bulan.
Data jadwal ruangan diambil lewat API dari sistem kampus.
API cuaca yang saya pakai diperbarui setiap satu jam.
Tim pengembang sedang memperbaiki API pemesanan ruang rapat.
Aplikasi kantin terhubung ke kasir melalui API khusus.
Dokumentasi API bisa Anda baca di portal pengembang.
Kalau API sedang lambat, jawaban saya juga ikut tertunda.
Silakan pindai kode QR di meja untuk mengisi buku tamu.
Kode QR pada undangan Anda akan diperiksa di pintu masuk.
Menu kantin bisa dilihat dengan memindai QR di setiap meja.
Tunjukkan QR tiket Anda kepada petugas di depan aula.
Kode QR ini hanya berlaku untuk satu kali masuk.
Saya bisa menampilkan QR untuk terhubung ke jaringan tamu.
Tempelkan kartu Anda ke pembaca NFC di samping pintu.
Fitur NFC di ponsel harus dinyalakan dulu sebelum menempel.
Kartu mahasiswa baru sudah dilengkapi cip NFC.
NFC di meja resepsionis bisa menyimpan kartu nama digital kami.
Pembayaran di kantin juga bisa dengan ponsel yang punya NFC.
Maaf, pembaca NFC di pintu ini sedang diperbaiki.
Tamu VIP dipersilakan menunggu di ruang khusus lantai dua.
Ruang VIP ada di ujung lorong sebelah kanan.
Area parkir VIP berada tepat di depan gedung utama.
Undangan VIP mendapat tempat duduk di barisan paling depan.
Saya akan mengantar Anda ke pintu masuk VIP sekarang.
VIP hari ini adalah rombongan dari kementerian.
Mohon tunjukkan kartu ID Anda kepada petugas keamanan.
ID pengunjung harus dipakai selama berada di dalam gedung.
Nomor ID pesanan Anda sudah saya kirim lewat pesan singkat.
Silakan masukkan ID dan kata sandi di layar ini.
Kartu ID sementara bisa diambil di meja resepsionis.
Tanpa ID yang sah, saya tidak bisa membukakan pintu laboratorium.
Tim IT sedang memperbaiki jaringan di lantai tiga.
Kalau komputer Anda bermasalah, silakan hubungi bagian IT.
Ruang IT berada di sebelah perpustakaan.
Petugas IT akan datang dalam sepuluh menit.
Kata sandi jaringan bisa diminta ke meja layanan IT.
Saya sudah melaporkan gangguan ini ke tim IT kampus.
Pembayaran di kantin bisa memakai QRIS.
QRIS toko ini ada di samping mesin kasir.
Silakan pindai QRIS di layar untuk membayar tiket.
Semua dompet digital yang mendukung QRIS bisa dipakai di sini.
Maaf, QRIS kami sedang gangguan, bisa bayar tunai dulu?
Donasi untuk acara ini juga bisa disalurkan lewat QRIS.
Bukti bayar QRIS Anda sudah masuk, terima kasih.
Kalau tidak bawa uang tunai, pakai QRIS saja ya.
Jadwal lengkapnya saya kirim dalam bentuk PDF.
PDF panduan tamu bisa diunduh lewat kode di meja.
Mohon unggah proposal Anda sebagai berkas PDF.
Berkas PDF itu terlalu besar untuk dikirim lewat surel.
Sertifikat peserta akan dikirim sebagai PDF minggu depan.
Saya bisa membacakan ringkasan dari PDF yang Anda kirim.
Selamat datang di kampus ITS Surabaya.
ITS memiliki banyak tim robot yang berprestasi.
Gedung ini adalah pusat riset robotika milik ITS.
Mahasiswa ITS sedang menyiapkan lomba tingkat nasional.
Perpustakaan ITS buka sampai malam selama masa ujian.
Saya dibuat oleh tim peneliti dari ITS.
Acara dimulai pukul sembilan WIB.
Pintu aula dibuka pukul tujuh tiga puluh WIB.
Rapat ditunda sampai pukul dua siang WIB.
Pendaftaran ditutup hari ini pukul empat sore WIB.
Semua jadwal di layar memakai waktu WIB.
Kereta terakhir berangkat pukul sepuluh malam WIB.
Mohon hadir lima belas menit sebelum pukul delapan WIB.
Siaran langsungnya tayang pukul satu siang WIB.
Sambungkan laptop ke proyektor dengan kabel HDMI.
Kabel HDMI cadangan ada di lemari ruang rapat.
HDMI di layar ini ada dua, pakai yang sebelah kiri.
Kalau gambarnya tidak muncul, coba cabut lalu pasang lagi kabel HDMI.
Saya bisa meminjamkan adaptor HDMI untuk laptop Anda.
Port HDMI proyektor aula sedang diperbaiki.
Kamera di ruang ini merekam dengan kualitas HD.
Siaran wisuda ditayangkan dalam format HD.
Layar HD di lobi menampilkan peta gedung.
Peserta dari Jayapura bergabung pukul sebelas WIT.
Siaran untuk wilayah timur dimulai pukul sembilan WIT.
Jadwal itu memakai waktu WIT, bukan waktu Surabaya.
Tim dari Makassar hadir daring pukul sepuluh WITA.
Pesawat ke Bali berangkat pukul dua siang WITA.
Mohon perhatikan, jadwal itu memakai waktu WITA.
Mohon siapkan KTP Anda untuk pendataan tamu.
KTP akan dikembalikan saat Anda meninggalkan gedung.
Tanpa KTP, kartu tamu tidak bisa dicetak.
Mahasiswa cukup menunjukkan KTM di pintu masuk.
KTM yang hilang bisa diurus di bagian akademik.
Tempelkan KTM Anda untuk meminjam buku.
Stan UMKM ada di halaman depan gedung.
Produk UMKM binaan kampus dijual di lantai satu.
Pelaku UMKM mendapat pelatihan gratis setiap bulan.
Klinik kampus menerima pasien BPJS.
Mohon bawa kartu BPJS saat berobat.
Petugas akan memeriksa kartu BPJS Anda lebih dulu.
Formulir ini meminta nomor NPWP Anda.
NPWP diperlukan untuk pencairan dana hibah.
Kalau belum punya NPWP, kolom itu boleh dikosongkan.
Perwakilan BUMN akan hadir di acara pembukaan.
Kampus bekerja sama dengan beberapa BUMN di bidang energi.
Beasiswa dari BUMN dibuka mulai bulan depan.
Tim PKM sedang menguji alatnya di laboratorium.
Proposal PKM dikumpulkan paling lambat hari Jumat.
Juara PKM tahun ini berasal dari jurusan teknik komputer.
Syarat beasiswa ini adalah IPK minimal tiga.
IPK Anda bisa dilihat di portal akademik.
Nilai IPK diperbarui setiap akhir semester.
Pengisian KRS dibuka mulai hari Senin.
KRS harus disetujui dosen wali sebelum batas waktu.
Perubahan KRS hanya bisa dilakukan minggu ini.
Ruang sidang FTEIC ada di lantai tiga.
Mahasiswa FTEIC berkumpul di aula barat.
Dekan FTEIC akan membuka acara pagi ini.
Rombongan dari UGM sudah tiba di lobi.
Peneliti UGM akan memaparkan hasil risetnya.
Tim robot UGM menjadi lawan kami di final.
Tamu dari ITB dijadwalkan datang siang ini.
Kami sering bertukar data riset dengan ITB.
Tim ITB membawa robot pemadam api.
Delegasi UI duduk di barisan kedua.
Dosen tamu dari UI mengajar hari Kamis.
Mahasiswa UI ikut dalam lomba ini.
Mesin tunai BNI ada di samping kantin.
Pembayaran bisa lewat transfer BNI.
Kantor kas BNI buka sampai sore.
Biaya pendaftaran dibayar lewat BRI.
Mesin tunai BRI berada di gedung rektorat.
Rekening BRI panitia tertera di undangan.
Transfer lewat BCA biasanya langsung masuk.
Mesin tunai BCA ada di dekat gerbang utama.
Bukti transfer BCA mohon dikirim ke panitia.
Halo, nama saya RAISA.
RAISA siap mengantar Anda ke ruang rapat.
Kalau butuh bantuan, panggil saja RAISA.
Aplikasi NABBIT bisa diunduh lewat kode di layar.
NABBIT membantu mencatat pengeluaran harian Anda.
Tim NABBIT membuka stan di lantai satu.
Robot BIMA sedang berlatih di lapangan.
BIMA adalah robot sepak bola kebanggaan kami.
Tim BIMA berangkat ke lomba minggu depan.
LUNA adalah robot penari dari tim kami.
Penampilan LUNA dimulai sebentar lagi.
Anda bisa berfoto dengan LUNA di panggung.
Tamu dari UNAIR sudah menunggu di lobi.
Kami bekerja sama dengan UNAIR di bidang kesehatan.
Dokter dari UNAIR akan memberi kuliah tamu.
Peneliti BRIN berkunjung ke laboratorium kami.
Riset ini didanai oleh BRIN.
Tim BRIN akan menilai hasil penelitian besok.
Mohon titipkan SIM Anda untuk meminjam kendaraan.
SIM dan kartu tamu ditukar di pos keamanan.
Tanpa SIM, mobil kampus tidak bisa dipinjam.
Komputer ini perlu tambahan RAM.
RAM laptop Anda sudah cukup untuk aplikasi ini.
Server kami memakai RAM yang sangat besar.
Sambungkan kabel LAN ke port di bawah meja.
Jaringan LAN di ruang ini lebih cepat daripada nirkabel.
Kabel LAN cadangan ada di laci.
EXPO teknologi digelar di halaman utama.
Stan kami di EXPO ada di dekat panggung.
Pengunjung EXPO boleh mencoba robotnya langsung.
SPARK adalah sistem pelaporan pintar buatan mahasiswa.
Tim SPARK menang lomba inovasi tahun ini.
Laporan kerusakan bisa dikirim lewat SPARK.
Sekretariat HIMATEKTRO ada di lantai dasar.
HIMATEKTRO mengadakan pelatihan robot akhir pekan ini.
Pengurus HIMATEKTRO sedang rapat di ruang sebelah.
"""

SENTENCES_EN = """
You can charge your phone using the USB port on this table.
Please plug your USB drive into the screen on the left.
A spare USB cable is in the top drawer.
Our research server uses a new GPU for training models.
The GPU in this lab runs almost all week.
Without a GPU, my speech recognition would be much slower.
The room schedule comes from the campus API.
Our developers are fixing the booking API today.
You can read the API guide on the developer portal.
Please scan the QR code on the table to sign in.
Show the QR code on your ticket at the entrance.
This QR code works only once.
Today's schedule is shown on the TV near the entrance.
You can watch TV while you wait.
The TV in the lobby is showing the campus profile.
Please connect your laptop with the HDMI cable.
A spare HDMI cable is in the meeting room cabinet.
The HDMI port on this projector is being repaired.
Welcome to the ITS campus in Surabaya.
I was built by a research team at ITS.
ITS has many award-winning robot teams.
I will send the full schedule as a PDF.
Please upload your proposal as a PDF file.
You can download the visitor guide as a PDF.
"""

# Added later (2026-11-16, readings confirmed by the user). Kept in separate blocks at the end so
# the clip numbers of the rows above never change.
SENTENCES_ID_2 = """
Kompetisi MAGE tahun ini diikuti ratusan tim.
Pendaftaran MAGE dibuka sampai akhir bulan.
Panitia MAGE berkumpul di aula timur.
Tim IRIS sedang menyiapkan robot sepak bola beroda.
Robot IRIS menjuarai lomba tingkat nasional.
Anda bisa melihat latihan tim IRIS di lapangan dalam.
Poin SKEM Anda bertambah setelah mengikuti acara ini.
Sertifikat ini bisa diklaim untuk SKEM.
Batas pengajuan SKEM adalah akhir semester.
CPU komputer ini cukup cepat untuk menyunting video.
Suhu CPU naik saat banyak program dibuka.
Laptop baru itu memakai CPU delapan inti.
Kipas di atas CPU perlu dibersihkan secara rutin.
Beban CPU server sedang tinggi siang ini.
Tanpa pendingin, CPU bisa mati mendadak.
Data riset disimpan di SSD supaya lebih cepat dibaca.
SSD laptop Anda hampir penuh.
Komputer laboratorium sudah diganti memakai SSD.
Harga SSD sekarang jauh lebih murah.
Berkas cadangan ada di SSD eksternal di lemari.
Memindahkan sistem ke SSD membuat komputer terasa baru.
Teknologi AI membantu saya memahami ucapan Anda.
AI di robot ini berjalan di server kampus.
Mahasiswa belajar dasar AI di semester lima.
Seminar tentang AI dimulai sore ini.
Jawaban saya dibuat oleh model AI lokal.
Riset AI di kampus ini berkembang pesat.
Lampu LED di lorong menyala otomatis saat ada orang.
Layar LED besar di aula menampilkan jadwal acara.
LED merah berarti baterai saya hampir habis.
Semua lampu gedung sudah diganti dengan LED.
Lampu LED di kepala saya berkedip saat mendengarkan.
Papan LED di depan menunjukkan nomor antrean.
PC di ruang ini boleh dipakai oleh tamu.
Silakan masuk ke PC dengan akun tamu.
Laboratorium punya tiga puluh PC untuk praktikum.
PC di meja ujung sedang diperbaiki.
Berkas itu tersimpan di PC ruang rapat.
Matikan PC sebelum meninggalkan laboratorium.
"""

SENTENCES_EN_2 = """
The CPU in this laptop is fast enough for video editing.
My answers come from a local AI model.
Research data is stored on an SSD for faster access.
The LED screen in the hall shows today's schedule.
Guests may use the PC in this room.
AI helps me understand what you say.
The red LED means my battery is low.
This PC is being repaired.
"""

REFS = ["wavs_v3/ref_01.wav", "wavs_v3/ref_02.wav", "wavs_v3/ref_03.wav", "wavs_v3/ref_04.wav"]


def rows():
    n = 0
    for lang, table, block in (("id", ID, SENTENCES_ID), ("en", EN, SENTENCES_EN),
                              ("id", ID, SENTENCES_ID_2), ("en", EN, SENTENCES_EN_2)):
        for text in filter(None, map(str.strip, block.splitlines())):
            tokens = re.findall(r"\b[A-Z]{2,}\b", text)
            assert tokens and all(t in table for t in tokens), text  # every capital token needs a reading
            n += 1
            row = {
                "audio": f"wavs_v3/acr_{n:03d}.wav",
                "text": text,
                "hint": "; ".join(f'"{t}" diucapkan "{table[t][0]}"' for t in dict.fromkeys(tokens)),
                "expect": {t: table[t][1] for t in tokens},
                "lang": lang,
            }
            if n % 5 in (1, 3):  # 40% with a reference clip, as the official fine-tuning guide advises
                row["ref_audio"] = REFS[n % 4]
            yield row


if __name__ == "__main__":
    out = list(rows())
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    per = {}
    for r in out:
        for t in r["expect"]:
            per[t] = per.get(t, 0) + 1
    print(len(out), "rows;", ", ".join(f"{t} {c}" for t, c in sorted(per.items(), key=lambda x: -x[1])))
