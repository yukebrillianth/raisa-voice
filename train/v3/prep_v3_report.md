# LoRA v3 data

## Old 600 clips

Dropped 18 (of them 2 in the v2 holdout, which is never trained and stays scored):

| row | audio | why | text |
|---|---|---|---|
| 144 | wavs/utt_145.wav | QRIS heard 'kyu ar ai es' | Kami menerima pembayaran dengan dompet digital apa pun yang mendukung QRIS standar. |
| 228 | wavs/utt_229.wav | QRIS heard 'ki ar ai es' | Denda keterlambatan pengembalian buku bisa dibayarkan langsung menggunakan QRIS di konter. |
| 273 | wavs/utt_274.wav | QA heard 'kyu ei' | Kalau ada bug atau tampilan yang rusak, tolong screenshot dan laporkan ke tim QA ya. |
| 287 | wavs/utt_288.wav | PDF heard 'pe de ef' | File presentasi format PowerPoint atau PDF bisa langsung diunggah lewat portal web kami. |
| 313 | wavs/utt_314.wav | GPU heard 'ge pe u' | Sistem pendingin server bekerja normal dengan suhu GPU terjaga di bawah enam puluh lima derajat. |
| 330 | wavs/utt_331.wav | QRIS heard 'kyu ar ai es' | Terima kasih sudah bertransaksi menggunakan QRIS, prosesnya cepat dan bebas uang kembalian. |
| 339 | wavs/utt_340.wav | NFC heard 'en ef ci' | Silakan ambil kartu nama digital kami dengan mendekatkan ponsel Anda ke tag NFC di meja. |
| 501 | wavs/utt_502.wav | WIB heard 'wip' | Acara seminar nasional teknologi hari ini dimulai tepat pukul 09.00 WIB di ruang utama. |
| 503 | wavs/utt_504.wav | WIT heard 'wip' | Sedangkan tim dari Jayapura dijadwalkan mengikuti sesi pembukaan pada pukul 11.00 WIT. |
| 521 | wavs/utt_522.wav | SKEM heard 'es ka em' | Sertifikat poin SKEM akan otomatis ditambahkan ke akun mahasiswa maksimal tiga hari kerja. |
| 522 | wavs/utt_523.wav | GPU heard 'ge pe u' | Server GPU lokal kita sudah terhubung ke API internal dengan latensi super rendah. |
| 547 | wavs/utt_548.wav | GB heard 'gi ga bait' | Prosesor komputer mini di robot ini berjalan pada frekuensi 4,2 GHz dengan RAM 32 GB. |
| 567 | wavs/utt_568.wav | WIB heard 'wib' | Silakan serahkan paket ini kepada RAISA di meja depan sebelum pukul 15.00 WIB. |
| 569 | wavs/utt_570.wav | MAGE heard 'meij' | Informasi lengkap mengenai kompetisi MAGE dapat Anda lihat pada layar di sebelah kiri. |
| 570 | wavs/utt_571.wav | IRIS heard 'i ris' | Tim robotika IRIS dari ITS sedang melakukan persiapan demo di ruang pameran utama. |
| 572 | wavs/utt_573.wav | GPU heard 'gi pi u' | Model bahasa lokal SENO sudah dijalankan langsung di server GPU kampus. |
| 584 | wavs/utt_585.wav | WIB heard 'wi i be' | Ibu DEWI selaku pembina acara akan memberikan sambutan pada pukul 09.30 WIB. |
| 597 | wavs/utt_598.wav | MAGE heard 'meij' | Jika ada pertanyaan untuk MAGE atau NABBIT, silakan tanyakan langsung kepada RAISA. |

Kept for training: 554 with their v2 hybrid transcripts (554 are identical to v3_text.v3(); the rest differ only where v2's units/letters were set by hand).

## New general clips (280 + 30 English)

Whisper large-v3-turbo, beam 5, transcribed as id and as en, the one closer to the script kept (1-s clips are detected as English), put through v3_text.v3() (digits, titles). CER over letters and digits (spaces and punctuation ignored) against the script: median 0.000, 90th percentile 0.016. Gemini acronym check on 32 clips that contain one.

Dropped 7:

| audio | why | script | Whisper heard |
|---|---|---|---|
| wavs_v3/v3_002.wav | CER 1.00 | Siap! | See ya! |
| wavs_v3/v3_055.wav | IRIS heard 'i ris' | Tim robotika IRIS ITS merupakan tim riset mahasiswa yang berfokus pada pengembangan robot sepak bola beroda otonom untuk kompetisi nasional dan internasional. Mereka rutin melakukan uji coba navigasi, visi komputer, dan komunikasi antarrbot di ruang Teaching Factory. Apakah Anda tertarik melihat jadwal demo robot mereka minggu ini? | Tim Robotika Iris ITS merupakan tim riset mahasiswa yang berfokus pada pengembangan robot sepak bola beroda otonom untuk kompetisi nasional dan internasional. Mereka rutin melakukan uji coba navigasi, visi komputer, dan komunikasi antarobot di ruang Teaching Factory. Apakah Anda tertarik melihat jadwal demo robot mereka minggu ini? |
| wavs_v3/v3_059.wav | QRIS heard 'ki ar ai es' | Kantin Pusat ITS menawarkan beragam pilihan makanan dan minuman dengan harga terjangkau mulai dari belasan ribu rupiah. Pembayaran di sebagian besar gerai kini sudah mendukung transaksi non-tunai menggunakan QRIS agar lebih cepat dan praktis. Apakah Anda butuh rekomendasi tempat makan yang masih sepi siang ini? | Kantin pusat ITS menawarkan beragam pilihan makanan dan minuman dengan harga terjangkau mulai dari belasan ribu rupiah. Pembayaran di sebagian besar gerai kini sudah mendukung transaksi non-tunai menggunakan QRIS agar lebih cepat dan praktis. Apakah Anda butuh rekomendasi tempat makan yang masih sepis yang ini? |
| wavs_v3/v3_074.wav | CER 0.57; words -mereka menyediakan pendampingan bisnis ruang kerja bersama serta akses jejaring ke berbagai investor teknologi apakah anda sedang mengembangkan produk rintisan yang ingin didaftarkan ke program inkubasi | Pusat Inkubasi Bisnis dan Teknologi membantu mahasiswa serta alumni mengubah hasil riset laboratorium menjadi perusahaan rintisan yang siap masuk ke pasar. Mereka menyediakan pendampingan bisnis, ruang kerja bersama, serta akses jejaring ke berbagai investor teknologi. Apakah Anda sedang mengembangkan produk rintisan yang ingin didaftarkan ke program inkubasi? | Pusat Inkubasi Bisnis dan Teknologi membantu mahasiswa serta alumni mengubah hasil riset laboratorium menjadi perusahaan rintisan yang siap masuk ke pasar. |
| wavs_v3/v3_086.wav | QR heard 'ki ar' | Semua perangkat pengisi daya kendaraan listrik di area parkir barat mendukung pengisian cepat standar untuk mobil maupun sepeda motor listrik. Pengguna cukup memindai kode QR pada tiang pengisi daya untuk memantau status baterai lewat ponsel. Apakah Anda datang menggunakan kendaraan listrik hari ini? | Semua perangkat pengisi daya kendaraan listrik di area parkir barat mendukung pengisian cepat standar untuk mobil maupun sepeda motor listrik. Pengguna cukup memindai kode QR pada tiang pengisi daya untuk memantau status baterai lewat ponsel. Apakah Anda datang menggunakan kendaraan listrik hari ini? |
| wavs_v3/v3_256.wav | IRIS heard 'i ris' | Doktor Insinyur Muhtadin dan Profesor Insinyur Rudy memberikan arahan langsung kepada tim robotika IRIS. | Doktor Insinyur Muhtadin dan Profesor Insinyur Rudy memberikan arahan langsung kepada tim robotika Iris. |
| wavs_v3/v3_280.wav | FTEIC heard 'ef te e i ka' | Terima kasih kepada Profesor Doktor Insinyur Diana Purwitasari dan seluruh dosen FTEIC atas dukungannya hari ini. | Terima kasih kepada Profesor Doktor Insinyur Diana Purwitasari dan seluruh dosen FTEIK atas dukungannya hari ini. |

Kept with CER > 0.05 (Whisper slips on names and loanwords, listed for the record):

| audio | CER | script | Whisper heard |
|---|---|---|---|
| wavs_v3/v3_030.wav | 0.06 | Silakan duduk dulu. | Silahkan duduk dulu. |
| wavs_v3/v3_033.wav | 0.14 | Jangan khawatir. | Jangan halatir. |
| wavs_v3/v3_140.wav | 0.06 | Wah, selamat ya atas pencapaiannya! Tim kalian memang luar biasa kompak sih. | Wah, selamat ya atas luar pencapaiannya. Tim kalian memang luar biasa kompak sih. |
| wavs_v3/v3_141.wav | 0.06 | Eh, kalau boleh tahu, Anda ke sininya naik kendaraan pribadi atau commuter line? | Eh, kalau boleh tahu, Anda kesininya naik kendaraan pribadi atau komuter lain. |

## Acronym set

251 verified of 258; judge holdout 19 (one Indonesian clip each of AI, API, CPU, GPU, HDMI, ID, IT, ITS, LED, NFC, PC, PDF, QR, QRIS, SSD, TV, USB, VIP, WIB); trained 232.

## Training set

| part | clips | minutes | with ref_audio | longest s |
|---|---|---|---|---|
| old 600 | 554 | 60.5 | 222 (40%) | 10.5 |
| new general | 303 | 53.3 | 122 (40%) | 28.6 |
| acronym | 232 | 17.2 | 95 (41%) | 10.5 |
| total | 1089 | 131.0 | 439 (40%) | 28.6 |
| holdout | 49 | 4.9 | 18 (37%) | 9.7 |

Transcripts changed by v3_text.v3() (new clips): 45

- `Tim dari Makassar hadir daring pukul sepuluh WITA.` -> `Tim dari Makassar hadir daring pukul sepuluh Wita.`
- `Pesawat ke Bali berangkat pukul dua siang WITA.` -> `Pesawat ke Bali berangkat pukul dua siang Wita.`
- `Mohon perhatikan, jadwal itu memakai waktu WITA.` -> `Mohon perhatikan, jadwal itu memakai waktu Wita.`
- `Halo, nama saya RAISA.` -> `Halo, nama saya Raisa.`
- `RAISA siap mengantar Anda ke ruang rapat.` -> `Raisa siap mengantar Anda ke ruang rapat.`
- `Kalau butuh bantuan, panggil saja RAISA.` -> `Kalau butuh bantuan, panggil saja Raisa.`
- `Aplikasi NABBIT bisa diunduh lewat kode di layar.` -> `Aplikasi Nabbit bisa diunduh lewat kode di layar.`
- `NABBIT membantu mencatat pengeluaran harian Anda.` -> `Nabbit membantu mencatat pengeluaran harian Anda.`
- `Tim NABBIT membuka stan di lantai satu.` -> `Tim Nabbit membuka stan di lantai satu.`
- `Robot BIMA sedang berlatih di lapangan.` -> `Robot Bima sedang berlatih di lapangan.`
- `BIMA adalah robot sepak bola kebanggaan kami.` -> `Bima adalah robot sepak bola kebanggaan kami.`
- `Tim BIMA berangkat ke lomba minggu depan.` -> `Tim Bima berangkat ke lomba minggu depan.`
- `LUNA adalah robot penari dari tim kami.` -> `Luna adalah robot penari dari tim kami.`
- `Penampilan LUNA dimulai sebentar lagi.` -> `Penampilan Luna dimulai sebentar lagi.`
- `Anda bisa berfoto dengan LUNA di panggung.` -> `Anda bisa berfoto dengan Luna di panggung.`
- `Tamu dari UNAIR sudah menunggu di lobi.` -> `Tamu dari Unair sudah menunggu di lobi.`
- `Kami bekerja sama dengan UNAIR di bidang kesehatan.` -> `Kami bekerja sama dengan Unair di bidang kesehatan.`
- `Dokter dari UNAIR akan memberi kuliah tamu.` -> `Dokter dari Unair akan memberi kuliah tamu.`
- `Peneliti BRIN berkunjung ke laboratorium kami.` -> `Peneliti Brin berkunjung ke laboratorium kami.`
- `Riset ini didanai oleh BRIN.` -> `Riset ini didanai oleh Brin.`
- `Tim BRIN akan menilai hasil penelitian besok.` -> `Tim Brin akan menilai hasil penelitian besok.`
- `Mohon titipkan SIM Anda untuk meminjam kendaraan.` -> `Mohon titipkan Sim Anda untuk meminjam kendaraan.`
- `SIM dan kartu tamu ditukar di pos keamanan.` -> `Sim dan kartu tamu ditukar di pos keamanan.`
- `Tanpa SIM, mobil kampus tidak bisa dipinjam.` -> `Tanpa Sim, mobil kampus tidak bisa dipinjam.`
- `Komputer ini perlu tambahan RAM.` -> `Komputer ini perlu tambahan Ram.`
- `RAM laptop Anda sudah cukup untuk aplikasi ini.` -> `Ram laptop Anda sudah cukup untuk aplikasi ini.`
- `Server kami memakai RAM yang sangat besar.` -> `Server kami memakai Ram yang sangat besar.`
- `Sambungkan kabel LAN ke port di bawah meja.` -> `Sambungkan kabel Lan ke port di bawah meja.`
- `Jaringan LAN di ruang ini lebih cepat daripada nirkabel.` -> `Jaringan Lan di ruang ini lebih cepat daripada nirkabel.`
- `Kabel LAN cadangan ada di laci.` -> `Kabel Lan cadangan ada di laci.`
- `EXPO teknologi digelar di halaman utama.` -> `Expo teknologi digelar di halaman utama.`
- `Stan kami di EXPO ada di dekat panggung.` -> `Stan kami di Expo ada di dekat panggung.`
- `Pengunjung EXPO boleh mencoba robotnya langsung.` -> `Pengunjung Expo boleh mencoba robotnya langsung.`
- `SPARK adalah sistem pelaporan pintar buatan mahasiswa.` -> `Spark adalah sistem pelaporan pintar buatan mahasiswa.`
- `Tim SPARK menang lomba inovasi tahun ini.` -> `Tim Spark menang lomba inovasi tahun ini.`
- `Laporan kerusakan bisa dikirim lewat SPARK.` -> `Laporan kerusakan bisa dikirim lewat Spark.`
- `Sekretariat HIMATEKTRO ada di lantai dasar.` -> `Sekretariat Himatektro ada di lantai dasar.`
- `HIMATEKTRO mengadakan pelatihan robot akhir pekan ini.` -> `Himatektro mengadakan pelatihan robot akhir pekan ini.`
- `Pengurus HIMATEKTRO sedang rapat di ruang sebelah.` -> `Pengurus Himatektro sedang rapat di ruang sebelah.`
- `Kompetisi MAGE tahun ini diikuti ratusan tim.` -> `Kompetisi Mage tahun ini diikuti ratusan tim.`
