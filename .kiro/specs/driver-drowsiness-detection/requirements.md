# Requirements Document

## Introduction

Bu doküman, "Sürücünün Uyku ve Yorgunluk Durumu Tespiti" bitirme projesinin
gereksinimlerini tanımlar. Sistem, YOLOv8 (Ultralytics) tabanlı bir görüntü
işleme uygulaması olarak; sürücünün gözlerinin açık veya kapalı olduğunu ve
ağzının esneme durumunda olup olmadığını kameradan alınan canlı görüntüler
veya statik fotoğraflar üzerinde tespit eder. Sistem, gözlerin belirli bir
süreden uzun kapalı kalması veya kısa zaman aralığında art arda esneme
gözlemlenmesi durumunda sürücüyü uyarır.

Geliştirme önce Windows üzerinde Python sanal ortamında (venv) yapılır,
ardından sistem Jetson Nano üzerine taşınır ve gömülü donanımda canlı
çalışacak şekilde optimize edilir. Mevcut veri seti `eye_yawn_dataset/`
klasöründe bulunan ve `Closed`, `Open`, `no_yawn`, `yawn` alt klasörlerine
ayrılmış görüntülerden oluşan klasör tabanlı bir sınıflandırma veri setidir.

## Kapsam

Bu spesifikasyon, bitirme projesi seviyesinde sade, modüler ve
anlaşılabilir bir uygulama hedefler. Aşağıdaki konular kapsam dışıdır:

- Yüz tanıma (kim olduğunu belirleme)
- Birden fazla sürücünün aynı kareyi paylaştığı senaryolar
- Araç içi CAN-Bus veya başka donanım ile entegrasyon
- Bulut tabanlı veri toplama veya uzaktan izleme

## Glossary

- **Sistem**: Sürücü Uyku ve Yorgunluk Tespit uygulamasının tamamı.
- **Veri_Hazirlayici**: `eye_yawn_dataset/` klasör yapısını eğitim
  formatına dönüştüren bileşen.
- **Egitici**: YOLOv8 modelini eğiten ve ağırlık dosyası üreten bileşen
  (`src/train.py`).
- **Tahminci**: Eğitilmiş model ile tek bir görüntü veya kare üzerinde
  sınıflandırma yapan bileşen.
- **Kamera_Yakalayicisi**: Webcam veya CSI kameradan kare okuyan bileşen
  (`src/webcam_detect.py` içinde).
- **Uyari_Mantigi**: Tahmin sonuçlarını zaman içinde takip ederek uyarı
  üreten bileşen (`src/alert_logic.py`).
- **Sesli_Uyarici**: Uyarı durumlarında ses çalan opsiyonel bileşen.
- **Yapilandirma**: Eşik değerleri ve parametreleri tutan dosya/yapı
  (örneğin `config.yaml` veya `utils.py` içinde sabitler).
- **Goz_Durumu**: Bir karedeki gözün durumu; `Open` veya `Closed`
  değerlerinden birini alır.
- **Agiz_Durumu**: Bir karedeki ağzın durumu; `yawn` veya `no_yawn`
  değerlerinden birini alır.
- **Kapali_Goz_Suresi**: Gözün kesintisiz kapalı kaldığı süre (saniye).
- **Esneme_Sayaci**: Belirli bir zaman penceresi içinde gözlemlenen
  esneme sayısı.
- **Uyari**: Sürücüye yönelik metinsel ve/veya sesli ikaz mesajı.
- **Hedef_Donanim**: Geliştirme PC'si (Windows) ve dağıtım hedefi olan
  Jetson Nano kartı.

## Requirements

### Requirement 1: Veri Seti Hazırlığı

**User Story:** Bir geliştirici olarak, klasör tabanlı görüntü veri setini
YOLOv8 ile eğitilebilir bir formata dönüştürmek istiyorum, ki mevcut
`eye_yawn_dataset/` veri setini sıfırdan etiketlemek zorunda kalmadan
kullanabileyim.

#### Acceptance Criteria

1. WHEN Veri_Hazirlayici çalıştırıldığında, THE Veri_Hazirlayici SHALL
   `eye_yawn_dataset/train` ve `eye_yawn_dataset/test` klasörlerinin
   altında `Closed`, `Open`, `no_yawn` ve `yawn` adında tam olarak dört
   alt klasörün varlığını doğrulamalı ve bu dört klasörü sınıf olarak
   tanımalıdır.
2. WHEN sınıf klasörleri doğrulandığında, THE Veri_Hazirlayici SHALL her
   sınıf klasöründeki `.jpg`, `.jpeg` ve `.png` uzantılı görüntü
   dosyalarını okumalı; bu uzantılar dışındaki dosyaları yok saymalıdır.
3. WHEN veri seti dönüşümü tetiklendiğinde, THE Veri_Hazirlayici SHALL
   veri setini YOLOv8 sınıflandırma (classification) formatına uygun
   olacak şekilde, kök altında `train/<sınıf_adı>/` ve
   `test/<sınıf_adı>/` alt dizinleri içeren tek ve deterministik bir
   dizin yapısı üretmelidir; nesne tespiti (detection) formatına
   dönüşüm yapılmamalıdır.
4. WHEN dizin yapısı üretildikten sonra, THE Veri_Hazirlayici SHALL
   üretilen veri setinin kök yolunu, dört sınıf adını ve her sınıfa
   karşılık gelen 0 ile 3 arasındaki tamsayı sınıf indekslerini içeren
   bir `data.yaml` dosyasını üretilen veri seti kök dizinine yazmalıdır.
5. IF beklenen alt klasörlerden (`Closed`, `Open`, `no_yawn`, `yawn`)
   en az biri `train` veya `test` altında bulunamazsa, THEN THE
   Veri_Hazirlayici SHALL eksik klasörün tam yolunu belirten bir hata
   mesajı vermeli, herhangi bir çıktı dizini veya `data.yaml` dosyası
   oluşturmamalı ve sıfırdan farklı bir çıkış koduyla sonlanmalıdır.
6. IF beklenen alt klasörlerden herhangi biri sıfır geçerli görüntü
   dosyası içeriyorsa, THEN THE Veri_Hazirlayici SHALL ilgili sınıf
   adını ve boş klasörün yolunu belirten bir hata mesajı vermeli ve
   sıfırdan farklı bir çıkış koduyla sonlanmalıdır.
7. WHEN dönüşüm başarıyla tamamlandığında, THE Veri_Hazirlayici SHALL
   terminale her bir sınıf için sınıf adını ve geçerli görüntü sayısını
   ayrı satırlarda, ayrıca `train` ve `test` bölümlerindeki toplam
   görüntü sayısını ve genel toplam görüntü sayısını yazdırmalıdır.

### Requirement 2: Model Eğitimi

**User Story:** Bir geliştirici olarak, hazırlanmış veri seti üzerinde
YOLOv8 modeli eğitmek istiyorum, ki sürücünün göz ve ağız durumunu
tahmin edebilen bir ağırlık dosyası elde edebileyim.

#### Acceptance Criteria

1. THE Egitici SHALL Ultralytics YOLOv8 kütüphanesini kullanarak model
   eğitimini başlatmalı ve eğitim sürecini bu kütüphanenin sağladığı
   API üzerinden yürütmelidir.
2. THE Egitici SHALL eğitim için kullanılan model boyutunu (varsayılan
   `yolov8n`, izin verilen değerler `yolov8n`, `yolov8s`, `yolov8m`,
   `yolov8l`, `yolov8x`), epoch sayısını (varsayılan `50`, izin verilen
   aralık `1`-`500`), görüntü boyutunu (varsayılan `640`, izin verilen
   aralık `320`-`1280`, `32`'nin katı) ve batch boyutunu (varsayılan
   `16`, izin verilen aralık `1`-`128`) yapılandırma dosyasından veya
   komut satırı argümanlarından okumalı; komut satırı argümanları
   yapılandırma dosyasındaki değerleri ezmelidir.
3. IF okunan parametrelerden herhangi biri tanımlı izin verilen aralığın
   veya değer kümesinin dışında ise, THEN THE Egitici SHALL hangi
   parametrenin geçersiz olduğunu, alınan değeri ve beklenen aralığı
   belirten bir hata mesajı vererek eğitimi başlatmamalıdır.
4. WHEN eğitim başarıyla tamamlandığında, THE Egitici SHALL doğrulama
   setinde en yüksek `accuracy` (sınıflandırma) veya `mAP@0.5`
   (nesne tespiti) değerine ulaşan kontrol noktasına ait ağırlık
   dosyasını `models/` klasörü altında sabit bir dosya adıyla
   (`best.pt`) kaydetmelidir.
5. IF `models/` klasörü mevcut değilse, THEN THE Egitici SHALL ağırlık
   dosyasını kaydetmeden önce bu klasörü oluşturmalıdır.
6. WHEN eğitim tamamlandığında, THE Egitici SHALL doğrulama seti
   üzerinde hesaplanan `accuracy` (sınıflandırma için) veya
   `mAP@0.5`, `mAP@0.5:0.95`, `precision`, `recall` (tespit için)
   metriklerini terminale tek seferde yazdırmalıdır.
7. IF veri seti kök yolu veya `data.yaml` dosyası belirtilen konumda
   bulunamazsa, THEN THE Egitici SHALL hangi yolun veya dosyanın eksik
   olduğunu içeren bir hata mesajı yazdırmalı, eğitimi başlatmamalı ve
   `models/` klasörü altındaki mevcut ağırlık dosyalarını
   değiştirmemelidir.
8. WHILE eğitim sürmektedir, THE Egitici SHALL her epoch tamamlandığında
   güncel epoch numarasını (`mevcut/toplam` biçiminde), eğitim kaybını
   (loss), doğrulama metriğini ve geçen süreyi terminale yazdırmalıdır.

### Requirement 3: Statik Görüntü Üzerinde Tahmin

**User Story:** Bir geliştirici olarak, internetten indirilmiş veya
test setinden seçilmiş bir fotoğraf üzerinde modelin tahminini görmek
istiyorum, ki modelin doğru çalışıp çalışmadığını canlı kameraya
geçmeden önce doğrulayabileyim.

#### Acceptance Criteria

1. WHEN bir görüntü dosyası yolu komut satırı argümanı olarak
   verildiğinde, THE Tahminci SHALL `models/` klasöründeki eğitilmiş
   ağırlık dosyasını yüklemeli, görüntüyü modele beslemeli ve `Closed`,
   `Open`, `no_yawn`, `yawn` sınıfları arasından en yüksek güven
   skoruna sahip sınıfı tahmin sonucu olarak belirlemelidir.
2. WHEN bir görüntü üzerindeki tahmin tamamlandığında, THE Tahminci
   SHALL `Closed`, `Open`, `no_yawn`, `yawn` sınıflarının her biri
   için `0.00` ile `1.00` aralığında iki ondalık basamaklı bir güven
   skorunu ve seçilen üst sınıfın adını terminale yazdırmalıdır;
   tahmin tamamlanmadan hiçbir güven skoru raporlanmamalıdır.
3. THE Tahminci SHALL üst sınıf adını ve buna karşılık gelen güven
   skorunu görüntü üzerine yazı olarak bindirmeli ve kullanıcı
   tarafından belirtilen çıktı moduna göre bindirilmiş görüntüyü ya
   ekranda göstermeli ya da verilen yola kaydetmeli; her iki mod da
   belirtilmediğinde varsayılan olarak görüntüyü ekranda göstermelidir.
4. IF verilen yol mevcut değilse, `.jpg`, `.jpeg`, `.png` veya `.bmp`
   uzantılarından birine sahip değilse ya da dosya geçerli bir görüntü
   olarak yüklenemiyorsa, THEN THE Tahminci SHALL hatanın nedenini ve
   verilen yolu açıkça belirten bir hata mesajı yazmalı ve tahmin
   yapmadan programdan çıkmalıdır.
5. IF Yapilandirma'daki `model_path` ile belirtilen ağırlık dosyası
   `models/` klasöründe bulunamazsa, THEN THE Tahminci SHALL aranan
   ağırlık dosyasının tam yolunu içeren açıklayıcı bir hata mesajı
   yazmalı ve tahmin yapmadan programdan çıkmalıdır.

### Requirement 4: Webcam ile Canlı Tespit

**User Story:** Bir sürücü olarak, bilgisayarın webcam'i veya Jetson
Nano'ya bağlı kamera üzerinden canlı tespit yapılmasını istiyorum, ki
gerçek zamanlı olarak göz ve ağız durumum değerlendirilebilsin.

#### Acceptance Criteria

1. WHEN canlı tespit modu başlatıldığında, THE Kamera_Yakalayicisi
   SHALL varsayılan kamera cihazını (`device=0`) `5` saniye içinde
   açmalı ve PC üzerinde en az `15` FPS hızında sürekli olarak kare
   okumalıdır.
2. WHERE Yapilandirma'da farklı bir kamera indeksi (`0`-`9` arası tam
   sayı) veya kaynak yolu (geçerli dosya yolu ya da RTSP/USB cihaz
   tanımlayıcısı, en fazla `260` karakter) belirtilmişse, THE
   Kamera_Yakalayicisi SHALL bu kaynağı kullanmalı ve kaynak
   geçersizse `5` saniye içinde hata bildirip çıkmalıdır.
3. WHEN her kare okunduğunda, THE Tahminci SHALL bu kare üzerinde
   tahmin yapmalı ve `Goz_Durumu` ile `Agiz_Durumu` değerlerini, kare
   yakalama anından itibaren en geç `200` ms içinde Uyari_Mantigi
   bileşenine iletmelidir.
4. WHEN bir kare için tahmin tamamlandığında, THE Sistem SHALL
   `Goz_Durumu`, `Agiz_Durumu` etiketlerini ve aktif uyarı(lar)ı kare
   üzerinde, kare yüksekliğinin en az `%3`'ü büyüklüğünde yazı tipiyle
   ve arka plandan ayırt edilebilecek kontrastlı renkle, kare alanının
   en fazla `%25`'ini kaplayacak şekilde göstermelidir.
5. IF kamera `5` saniye içinde açılamazsa, THEN THE Kamera_Yakalayicisi
   SHALL hatayı tanımlayan bir mesaj çıktılamalı, açık olan tüm kamera
   kaynaklarını serbest bırakmalı ve sıfır olmayan bir çıkış koduyla
   programdan çıkmalıdır.
6. IF ardışık `30` kare okunamazsa veya kamera akışı `3` saniye boyunca
   yeni kare üretmezse, THEN THE Kamera_Yakalayicisi SHALL hata mesajı
   çıktılamalı, kamera kaynağını serbest bırakmalı ve programdan
   çıkmalıdır.
7. WHEN kullanıcı `q` tuşuna basarsa, THE Sistem SHALL `1` saniye
   içinde kare okumayı durdurmalı, kamera kaynağını ve görüntüleme
   penceresini serbest bırakmalı ve sıfır çıkış koduyla uygulamadan
   çıkmalıdır.

### Requirement 5: Uyku Uyarı Mantığı (Kapalı Göz Süresi)

**User Story:** Bir sürücü olarak, gözlerimi belirli bir süreden uzun
süre kapalı tuttuğumda uyarı almak istiyorum, ki uyuya kalmadan önce
uyandırılabileyim.

#### Acceptance Criteria

1. WHEN bir kare için `Goz_Durumu` `Closed` olarak tahmin edildiğinde
   ve önceki karede `Goz_Durumu` `Open` ise veya bu kare sistemin
   işlediği ilk geçerli `Goz_Durumu` tahmini ise, THE Uyari_Mantigi
   SHALL `eye_closed_start_time` değerini saniye cinsinden geri
   sarılmayan (monotonik) sistem zamanı olarak kaydetmelidir.
2. WHILE ardışık karelerde `Goz_Durumu` `Closed` olarak tahmin
   edilmektedir, THE Uyari_Mantigi SHALL her yeni kare için
   `Kapali_Goz_Suresi` değerini güncel monotonik zaman ile
   `eye_closed_start_time` arasındaki fark olarak saniye cinsinden ve
   negatif olmayan biçimde hesaplamalıdır.
3. WHEN `Kapali_Goz_Suresi` değeri Yapilandirma'daki
   `closed_eye_duration` eşiğine eşit veya bu eşikten büyük olduğunda
   ve aktif uyukluyor uyarı durumu yokken, THE Uyari_Mantigi SHALL
   "UYARI: Sürücü uyukluyor olabilir!" mesajını bir kez üretmeli ve
   aktif uyukluyor uyarı durumunu işaretlemelidir.
4. WHEN bir kare için `Goz_Durumu` `Open` olarak tahmin edildiğinde,
   THE Uyari_Mantigi SHALL `eye_closed_start_time` değerini
   sıfırlamalı, `Kapali_Goz_Suresi` değerini `0.0` saniye yapmalı ve
   aktif uyukluyor uyarı durumunu temizlemelidir.
5. THE Yapilandirma SHALL `closed_eye_duration` parametresini saniye
   cinsinden, `0.5` ile `10.0` arasında (uç değerler dahil) bir kayan
   noktalı sayı olarak tanımlamalı ve varsayılan değerini `2.0` olarak
   ayarlamalıdır.
6. WHILE aktif uyukluyor uyarı durumu işaretliyken ve ardışık karelerde
   `Goz_Durumu` `Closed` olarak tahmin edilmeye devam ederken, THE
   Uyari_Mantigi SHALL aynı uyukluyor uyarı mesajını tekrar
   üretmemelidir; ancak `Goz_Durumu` en az bir kare `Open` olarak
   tahmin edildikten sonra `Kapali_Goz_Suresi` yeniden
   `closed_eye_duration` eşiğine ulaşırsa, THE Uyari_Mantigi SHALL yeni
   bir uyukluyor uyarı mesajı üretmelidir.
7. IF bir kare için `Goz_Durumu` tahmini elde edilemezse (eksik veya
   tanımsız), THEN THE Uyari_Mantigi SHALL o kareyi `Kapali_Goz_Suresi`
   hesaplamasında kullanmamalı ve `eye_closed_start_time` ile aktif
   uyukluyor uyarı durumu değerlerini değiştirmemelidir.

### Requirement 6: Yorgunluk Uyarı Mantığı (Esneme Sayısı)

**User Story:** Bir sürücü olarak, kısa bir zaman aralığında art arda
esnediğimde uyarı almak istiyorum, ki yorgunluğumun farkına varıp mola
verebileyim.

#### Acceptance Criteria

1. WHEN bir kare için `Agiz_Durumu` `yawn` olarak tahmin edildiğinde
   ve önceki karede `Agiz_Durumu` `no_yawn` ise veya bu kare sistemin
   işlediği ilk geçerli `Agiz_Durumu` tahmini ise, THE Uyari_Mantigi
   SHALL geçerli monotonik zaman damgasını yeni bir esneme olayı
   olarak kaydetmelidir.
2. WHEN her yeni kare işlendiğinde, THE Uyari_Mantigi SHALL son
   `yawn_time_window` saniye içinde kaydedilmiş esneme olaylarını
   sayarak `Esneme_Sayaci` değerini güncellemeli ve daha eski olayları
   listeden çıkarmalıdır.
3. IF `Esneme_Sayaci` değeri Yapilandirma'daki `yawn_count` eşiğine
   eşit veya bu eşikten büyükse ve aktif yorgunluk uyarı durumu
   yoksa, THEN THE Uyari_Mantigi SHALL "UYARI: Sürücü yorgun
   olabilir!" mesajını bir kez üretmeli ve aktif yorgunluk uyarı
   durumunu işaretlemelidir.
4. WHEN aktif yorgunluk uyarı durumu işaretliyken `Esneme_Sayaci`
   değeri `yawn_count` eşiğinin altına düşerse, THE Uyari_Mantigi
   SHALL aktif yorgunluk uyarı durumunu temizlemelidir.
5. THE Yapilandirma SHALL `yawn_count` parametresini `1`-`20` arasında
   tam sayı olarak ve varsayılan değeri `3` olacak şekilde,
   `yawn_time_window` parametresini saniye cinsinden `10.0`-`600.0`
   arasında kayan noktalı sayı olarak ve varsayılan değeri `60.0`
   olacak şekilde tanımlamalıdır.
6. WHILE ardışık karelerde `Agiz_Durumu` kesintisiz `yawn` olarak
   kalmaya devam ederken, THE Uyari_Mantigi SHALL bu bloğu yalnızca
   tek bir esneme olarak saymalı; ağız `no_yawn` durumuna döndükten
   sonra tekrar `yawn` olduğunda yeni bir esneme olayı olarak
   sayılmalıdır.
7. IF bir kare için `Agiz_Durumu` tahmini elde edilemezse (eksik veya
   tanımsız), THEN THE Uyari_Mantigi SHALL o kareyi yeni bir esneme
   olayı olarak saymamalı ve önceki kare durum bilgisini
   değiştirmemelidir.

### Requirement 7: Sesli Uyarı (Opsiyonel)

**User Story:** Bir sürücü olarak, görsel uyarıya ek olarak sesli
uyarı da almak istiyorum, ki gözüm yolda iken bile uyarıyı fark
edebileyim.

#### Acceptance Criteria

1. WHERE Yapilandirma'da `enable_sound` parametresi `true` olarak
   ayarlanmışsa, WHEN bir uyarı mesajı üretildiğinde, THE Sesli_Uyarici
   SHALL uyarı üretiminden itibaren `500` ms içinde sesli uyarıyı
   çalmaya başlamalıdır.
2. WHERE Yapilandirma'da `enable_sound` parametresi `false` olarak
   ayarlanmışsa veya tanımlı değilse, THE Sesli_Uyarici SHALL hiçbir
   ses çalmamalı ve görsel uyarı akışını etkilememelidir.
3. WHERE çalışılan platform Windows ise, THE Sesli_Uyarici SHALL
   `winsound` veya `playsound` kütüphanelerinden birini kullanarak
   ses çalmalıdır.
4. WHERE çalışılan platform Jetson Nano (Linux) ise, THE Sesli_Uyarici
   SHALL `playsound` veya eşdeğer çapraz platform bir kütüphane
   kullanarak ses çalmalıdır.
5. THE Sesli_Uyarici SHALL tek bir uyarı için çalınan sesin süresini
   en fazla `3` saniye ile sınırlamalıdır.
6. IF ses dosyası bulunamazsa, dosya okunamaz formatta ise veya ses
   çalma sırasında bir hata oluşursa, THEN THE Sesli_Uyarici SHALL
   hatayı, hata türünü ve zaman damgasını içerecek şekilde log
   dosyasına yazmalı, görsel uyarı akışını engellememeli ve sistemin
   çalışmasını sonlandırmamalıdır.
7. WHILE aynı uyarı türü için bir ses çalınmaktayken, THE Sesli_Uyarici
   SHALL aynı uyarı türü için gelen yeni ses çalma taleplerini, mevcut
   ses çalma işlemi tamamlanana kadar yok saymalıdır.

### Requirement 8: Yapılandırılabilir Eşik Değerleri

**User Story:** Bir geliştirici olarak, eşik değerlerini ve çalışma
parametrelerini koddan ayrı bir yerde değiştirebilmek istiyorum, ki
test sırasında farklı senaryoları kolayca deneyebileyim.

#### Acceptance Criteria

1. THE Yapilandirma SHALL en az şu parametreleri ve geçerli değer
   aralıklarını içermelidir: `closed_eye_duration` (`0.5`-`10.0`
   saniye), `yawn_count` (`1`-`20` adet), `yawn_time_window`
   (`10.0`-`600.0` saniye), `confidence_threshold` (`0.0`-`1.0`),
   `camera_index` (`0`-`10` arası tam sayı), `enable_sound`
   (`true`/`false`), `frame_skip` (`0`-`30` arası tam sayı),
   `model_path` (en fazla `512` karakterlik dosya yolu dizgisi).
2. WHEN Sistem başlatıldığında, THE Sistem SHALL Yapilandirma
   değerlerini bir dosyadan (`config.yaml` veya benzeri) ya da merkezî
   bir Python modülünden bir kez okumalı ve okunan tüm parametreleri
   başlangıç loglarına yazmalıdır.
3. WHEN Yapilandirma dosyası değiştirildikten sonra Sistem yeniden
   başlatıldığında, THE Sistem SHALL kaynak kodu yeniden derlemeye
   gerek kalmadan yeni değerleri `5` saniye içinde yüklemeli ve
   kullanmaya başlamalıdır.
4. IF bir Yapilandirma parametresi eksikse, THEN THE Sistem SHALL o
   parametre için belgelenmiş varsayılan değeri kullanmalı, eksik
   parametre adını ve uygulanan varsayılan değeri uyarı seviyesinde
   loga yazmalı ve çalışmaya devam etmelidir.
5. IF bir Yapilandirma parametresi geçersiz bir değere sahipse
   (sayısal parametrede negatif değer, tanımlı aralık dışı değer,
   beklenen tipte olmayan değer veya `model_path` için var olmayan
   dosya), THEN THE Sistem SHALL başlatma işlemini durdurmalı,
   sıfırdan farklı bir çıkış kodu döndürmeli ve hatalı parametrenin
   adını, alınan değeri ve red sebebini içeren bir hata mesajı
   yayımlamalıdır.
6. IF Yapilandirma kaynağı (dosya veya modül) bulunamazsa ya da
   ayrıştırılamayan biçim hatası içeriyorsa, THEN THE Sistem SHALL
   başlatma işlemini durdurmalı, kaynağın konumunu ve hata sebebini
   belirten bir hata mesajı yayımlamalı ve hiçbir Yapilandirma
   değerini varsayılana sessizce düşürmemelidir.

### Requirement 9: Loglama ve Ekrana Bilgi Yazdırma

**User Story:** Bir geliştirici olarak, sistemin tahminlerini ve
uyarılarını terminalde ve dosyada görmek istiyorum, ki test ve hata
ayıklama sırasında neler olduğunu anlayabileyim.

#### Acceptance Criteria

1. WHEN bir uyarı mesajı üretildiğinde, THE Sistem SHALL bu mesajı,
   ISO 8601 formatında (`YYYY-MM-DD HH:MM:SS`) zaman damgası, log
   seviyesi (`INFO`/`WARNING`/`ERROR`) ve mesaj metni içerecek
   şekilde, üretilmesinden itibaren `100` ms içinde terminale
   yazdırmalıdır.
2. WHEN bir uyarı mesajı üretildiğinde, THE Sistem SHALL aynı mesajı,
   ISO 8601 formatında zaman damgası ve log seviyesi ile birlikte,
   üretilmesinden itibaren `500` ms içinde log dosyasına satır
   eklemeli (append) ve önceki kayıtların üzerine yazmamalıdır.
3. WHILE Sistem canlı tespit modunda çalışırken, THE Sistem SHALL son
   `1` saniyedeki ortalama FPS değerini, Yapilandirma'daki
   `fps_log_interval` parametresiyle belirlenen aralıkta (`1` ile `60`
   saniye arası, varsayılan `1` saniye), iki ondalık basamaklı sayı
   olarak terminale yazdırmalıdır.
4. IF Yapilandirma'daki `fps_log_interval` parametresi `1`-`60`
   saniye aralığının dışında ise, THEN THE Sistem SHALL varsayılan
   değer olan `1` saniyeyi kullanmalı ve terminale geçersiz değer
   kullanıldığını belirten bir uyarı mesajı yazdırmalıdır.
5. WHERE Yapilandirma'da `verbose` parametresi `true` ise, WHILE
   Sistem canlı tespit modunda çalışırken, THE Sistem SHALL her
   işlenen kare için kare numarası, `Goz_Durumu` etiketi,
   `Agiz_Durumu` etiketi ve her bir tahmin için `0.00` ile `1.00`
   arasında iki ondalık basamaklı güven skorunu terminale
   yazdırmalıdır.
6. IF log dosyası yazma işlemi başarısız olursa (dosya erişim hatası,
   disk dolu veya izin reddi), THEN THE Sistem SHALL terminale hata
   nedenini belirten bir uyarı mesajı yazdırmalı, log dosyasına yazma
   denemelerini her `30` saniyede bir tekrar denemeli ve canlı tespit
   işlemini durdurmamalıdır.
7. IF log dosyası boyutu yapılandırılabilir maksimum sınırı (`1` MB
   ile `100` MB arası, varsayılan `10` MB) aşarsa, THEN THE Sistem
   SHALL mevcut log dosyasını arşivleyip yeni bir log dosyası
   oluşturmalı ve canlı tespit işlemini kesintiye uğratmamalıdır.

### Requirement 10: Jetson Nano'ya Taşıma ve Performans Optimizasyonu

**User Story:** Bir geliştirici olarak, sistemi Jetson Nano üzerinde
çalıştırabilmek istiyorum, ki proje gömülü hedef donanım üzerinde de
canlı çalışabildiğini gösterebilsin.

#### Acceptance Criteria

1. THE Sistem SHALL Jetson Nano üzerinde Python `3.6` veya üstü ve
   projenin `requirements.txt` dosyasında sabitlenmiş Ultralytics
   YOLOv8 sürümü ile çalışacak şekilde paketlenmeli, tüm bağımlılıklar
   `requirements.txt` üzerinden kurulabilmelidir.
2. THE Sistem SHALL Jetson Nano üzerinde USB veya CSI kameradan en az
   `640x480` çözünürlükte ve en az `15` FPS hızında görüntü almayı
   desteklemelidir.
3. WHERE Hedef_Donanim Jetson Nano ise, THE Sistem SHALL varsayılan
   model olarak `yolov8n` kullanmalıdır.
4. WHERE Yapilandirma'da `export_format` parametresi `onnx` veya
   `tensorrt` olarak ayarlanmışsa, THE Sistem SHALL eğitilmiş `.pt`
   modelini bu formata dönüştürmeli ve canlı tespitte dönüştürülmüş
   modeli kullanmalıdır.
5. IF model dönüşümü başarısız olursa, THEN THE Sistem SHALL hata
   sebebini logmalı, orijinal `.pt` modeline geri dönmeli ve canlı
   tespiti durdurmamalıdır.
6. WHERE Yapilandirma'da `frame_skip` parametresi `n` (`1`-`10` arası
   tam sayı) olarak ayarlanmışsa, THE Kamera_Yakalayicisi SHALL her
   `n`. kareyi tahmin için iletmeli, aradaki kareleri yalnızca
   görüntülemelidir.
7. IF `frame_skip` değeri `1`-`10` aralığının dışındaysa, THEN THE
   Sistem SHALL varsayılan değer olan `1`'i kullanmalı ve terminale
   geçersiz değer kullanıldığını belirten bir uyarı mesajı
   yazdırmalıdır.
8. WHERE Yapilandirma'da `inference_resolution` parametresi
   tanımlanmışsa (`160`-`1280` piksel arası, `32`'nin katı), THE
   Tahminci SHALL kareleri tahmin öncesinde bu çözünürlüğe
   ölçeklendirmelidir.
9. THE Sistem SHALL Jetson Nano üzerinde canlı tespit sırasında son
   `30` saniyelik kayan pencerede ortalama FPS değerinin en az `5`
   FPS olmasını hedeflemeli ve bu değeri her `5` saniyede bir
   terminale raporlamalıdır.
10. IF Jetson Nano üzerinde ortalama FPS değeri `30` saniye boyunca
    `5` FPS'in altında kalırsa, THEN THE Sistem SHALL terminale ve
    log dosyasına performans uyarısı yazmalı ve `inference_resolution`
    ile `frame_skip` parametrelerinin ayarlanmasını öneren bir mesaj
    göstermelidir.

### Requirement 11: Proje Yapısı ve Bağımlılıklar

**User Story:** Bir geliştirici olarak, projeyi sade ve modüler bir
klasör yapısında istiyorum, ki bitirme projesi için sunum ve
değerlendirme yapması kolay olsun.

#### Acceptance Criteria

1. THE Sistem SHALL proje kök dizininde aşağıdaki klasör ve dosya
   yapısını eksiksiz bulundurmalıdır: `dataset/` klasörü (eğitim
   verisi için), `models/` klasörü (eğitilmiş model ağırlıkları için),
   `src/` klasörü altında `train.py`, `webcam_detect.py`,
   `alert_logic.py` ve `utils.py` dosyaları, kök dizinde
   `requirements.txt` ve `README.md` dosyaları.
2. IF listelenen klasör veya dosyalardan herhangi biri eksikse, THEN
   THE Sistem yapı doğrulama kontrolünde başarısız sayılmalı ve eksik
   öğenin adı raporlanmalıdır.
3. THE `requirements.txt` dosyası SHALL en az `ultralytics`,
   `opencv-python` ve sesli uyarı için seçilen ses kütüphanesini her
   biri için kesin sürüm sabitlemesi (`paket==majör.minör.yama`
   formatında) ile içermelidir.
4. THE `requirements.txt` dosyası SHALL desteklenen Python sürümünü
   (Python `3.8` ile `3.11` arası) belirten bir yorum satırı içermeli
   ve `pip install -r requirements.txt` komutu temiz bir sanal
   ortamda hatasız tamamlanmalıdır.
5. THE `README.md` SHALL Türkçe yazılmış ve aşağıdaki başlıkların
   tamamını ayrı bölümler olarak içeren bir doküman olmalıdır: (a)
   Kurulum adımları (sanal ortam oluşturma ve bağımlılık yükleme
   komutları), (b) Eğitim komutu (çalıştırılabilir komut satırı
   örneği), (c) Statik tahmin komutu (örnek görsel üzerinde), (d)
   Canlı tespit komutu (webcam ile), (e) Jetson Nano'ya taşıma
   adımları (en az `3` adım).
6. IF `README.md` içinde yukarıda sayılan bölümlerden (Kurulum,
   Eğitim, Statik Tahmin, Canlı Tespit, Jetson Nano Taşıma) herhangi
   biri eksikse, THEN THE doküman doğrulama kontrolü başarısız
   sayılmalı ve eksik bölümün adı raporlanmalıdır.

## Doğruluk Özellikleri (Correctness Properties)

Bu bölüm, Uyari_Mantigi ve Yapilandirma davranışı için property-based
test ile doğrulanabilecek özellikleri tanımlar. Görüntü işleme ve
model eğitimi bölümleri için statik örnek tabanlı testler yeterlidir;
aşağıdaki özellikler ise sentetik tahmin dizileri (örneğin
"Closed/Closed/Open/yawn/..." gibi) üzerinde çalıştırılabildiği için
PBT'ye uygundur.

### P1: Uyku uyarısı yalnızca eşik aşıldığında üretilir

Sentetik bir kare-tahmin dizisi içinde gözün kesintisiz `Closed`
kaldığı en uzun süre `closed_eye_duration` eşiğinin altında ise,
Uyari_Mantigi uyku uyarısı üretmez. Eşik aşıldığı her kesintisiz
`Closed` bloğu için tam olarak bir uyku uyarısı üretilir.

### P2: Açılan göz sayacı sıfırlar

Sentetik bir kare dizisinde herhangi bir noktada `Goz_Durumu` `Open`
olarak gözlemlenirse, ondan sonraki ilk `Closed` karesinde
`Kapali_Goz_Suresi` değeri sıfırdan başlamalıdır.

### P3: Esneme zaman penceresi metamorfik özelliği

Sentetik bir esneme olay dizisinde aynı olay dizisi `yawn_time_window`
süresini aşan bir kayma ile ötelenirse, üretilen yorgunluk
uyarılarının sayısı değişmemelidir; pencere küçültüldüğünde uyarı
sayısı azalmalı veya eşit kalmalı, büyütüldüğünde ise artmalı veya
eşit kalmalıdır.

### P4: Tek esneme tek olay sayılır

Ardışık `yawn` etiketli kareler tek bir blok oluşturduğu sürece,
`Esneme_Sayaci` bu bloğu yalnızca bir olay olarak saymalıdır;
`no_yawn` arasına girmeden uzayan bloklar ek esneme olarak
sayılmamalıdır.

### P5: Yapılandırma değerleri davranışı belirler

Aynı kare-tahmin dizisi farklı `closed_eye_duration` değerleriyle
çalıştırıldığında, eşik büyüdükçe üretilen uyku uyarısı sayısı
azalmalı veya eşit kalmalıdır (monotoniklik). Aynı şekilde
`yawn_count` eşiği büyüdükçe yorgunluk uyarısı sayısı azalmalı veya
eşit kalmalıdır.

### P6: Düşük güvenli kareler uyarı tetiklemez

Tahminler `confidence_threshold` değerinin altında olduğunda,
Uyari_Mantigi bu kareyi `Closed` veya `yawn` olarak saymamalı; düşük
güvenli kareler uyarı tetikleme yolunda yok sayılmalıdır.

### P7: Frame-skip uyarı zamanlamasını bozmamalı

Aynı sentetik dizi `frame_skip = 1` ve `frame_skip = n` (`n > 1`)
ile işlendiğinde, üretilen uyarıların zaman damgaları en fazla
`n / FPS` saniye farklı olmalı; uyarı sayısı ise tek bir uyarı
çözünürlüğü farkıyla aynı kalmalıdır.

## Süreç Notları

- Veri seti klasör tabanlıdır ve YOLOv8 **classification** modu ile
  doğrudan kullanılacaktır; bu projede detection (bounding-box)
  formatına dönüşüm yapılmayacaktır.
- Sesli uyarı opsiyoneldir ve `enable_sound` ile kapatılabilir
  olmalıdır.
- Performans hedefleri Jetson Nano için minimum eşiklerdir; PC
  üzerinde bu eşiklerin rahatça aşılması beklenir.
