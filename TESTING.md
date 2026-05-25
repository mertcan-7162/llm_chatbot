# TESTING — Test ve Doğrulama

## 1. Test Yaklaşımı

Bu projede test süreci iki ana aşamada yürütüldü:

1. **Metin çıkarma doğrulaması:** Yüklenen PDF veya görsel dosyadan metnin gerçekten doğru çıkarılıp çıkarılmadığını anlamak için `scripts/extract_text.py` aracı yazıldı.
2. **Uçtan uca chatbot testi:** OCR, chunking, embedding, ChromaDB retrieval ve LLM cevap üretimi birlikte test edildi.

İlk aşamada amaç, chatbot cevaplarına geçmeden önce sistemin en temel girdisi olan çıkarılmış metnin kalitesini gözlemlemekti. OCR çıktısı hatalıysa RAG pipeline'ının doğru cevap üretmesi de zorlaşır.

---

## 2. Metin Çıkarma Test Aracı

Metin çıkarma kalitesini bağımsız olarak test edebilmek için `scripts/extract_text.py` dosyası yazıldı.

Bu script:

- PDF ve görsel dosyaları destekler.
- Görsellerde PaddleOCR kullanır.
- PDF dosyalarında önce native text extraction dener, gerekirse OCR fallback kullanır.
- Çıktıyı `.txt` veya `.json` formatında kaydedebilir.
- Test konfigürasyonunu `scripts/extract_text.yaml` dosyasından okur.

Örnek kullanım:

```bash
python scripts/extract_text.py
```

Test edilen örnek dosyalar `example_input_files/` altında bulunur. Script çıktıları ise `scripts/outputs/` klasöründe tutulur.

---

## 3. OCR ve Text Extraction Testleri

### 3.1 Türkçe Vikipedi Görselleri

Türkçe OCR performansını ölçmek için Vikipedi sayfalarından alınan ekran görüntüleri kullanıldı:

- `example_input_files/ataturk.png`
- `example_input_files/fatih_sultan_mehmed.png`

Bu dosyalar `scripts/extract_text.py` ile işlendi ve sonuçlar şu dosyalara yazıldı:

- `scripts/outputs/ataturk_extracted_text.txt`
- `scripts/outputs/fatih_sultan_mehmed_extracted_text.txt`

Gözlem:

- Metinlerin büyük bölümü doğru çıkarıldı.
- Türkçe karakterler genel olarak korunabildi.
- Fatih Sultan Mehmed örneğinde tarih, kişi adı, yer adı ve olay bilgileri başarılı şekilde çıkarıldı.
- Atatürk örneğinde metin yoğun ve ekran görüntüsü daha karmaşık olmasına rağmen içerik büyük oranda okunabilir şekilde elde edildi.
- Küçük OCR hataları oluşsa da metnin genel anlamı bozulmadı.

Sonuç olarak Türkçe Vikipedi ekran görüntülerinde OCR başarımı yüksek gözlemlendi.

### 3.2 İngilizce Vikipedi Görseli

İngilizce OCR performansını test etmek için Donald Trump Vikipedi sayfasından alınan ekran görüntüsü kullanıldı:

- `example_input_files/Trump.png`

Çıktı:

- `scripts/outputs/Trump_extracted_text.txt`

Gözlem:

- İngilizce metin neredeyse tamamen doğru çıkarıldı.
- Cümle yapısı ve özel isimler büyük ölçüde korundu.
- OCR çıktısı RAG pipeline'ında kullanılabilecek kadar temizdi.

Bu test, sistemin yalnızca Türkçe değil İngilizce belgelerde de başarılı metin çıkarabildiğini gösterdi.

### 3.3 El Yazısı Testi

Daha zor bir OCR senaryosu olarak el yazısı içeren görsel test edildi:

- `example_input_files/el_yazisi.png`

Çıktı:

- `scripts/outputs/el_yazisi_extracted_text.txt`

Gözlem:

- Sistem el yazısını kısmen okuyabildi.
- Genel anlam bazı yerlerde korunmasına rağmen karakter hataları arttı.
- Basılı metinlere göre başarımın belirgin şekilde düştüğü görüldü.

Bu test, sistemin zorlu ve doğal olmayan inputlarda sınırlarını görmek için yapıldı. Sonuç olarak PaddleOCR'ın basılı/dijital metinlerde oldukça başarılı, el yazısında ise daha sınırlı olduğu gözlemlendi.

### 3.4 PDF Extraction Testi

PDF desteğini doğrulamak için örnek PDF dosyası test edildi:

- `example_input_files/mert_cv.pdf`

Çıktı:

- `scripts/outputs/mert_cv_extracted_text.txt`

Gözlem:

- PDF dijital metin katmanı içerdiği için native extraction kullanıldı.
- OCR'a gerek kalmadan metin hızlı ve başarılı şekilde çıkarıldı.
- Başlıklar, deneyim bilgileri ve eğitim bilgileri okunabilir şekilde elde edildi.

Bu test, sistemin PDF dosyalarında yalnızca OCR'a bağlı kalmadığını, mümkün olduğunda daha güvenilir olan native text extraction yolunu kullandığını doğruladı.

---

## 4. Uçtan Uca Chatbot Pipeline Testleri

Metin çıkarma testlerinden sonra sistem web arayüzü üzerinden uçtan uca test edildi.

Test edilen akış:

1. Dosya yükleme
2. OCR veya PDF extraction
3. Chunk oluşturma
4. Embedding üretme
5. ChromaDB'ye kayıt
6. Kullanıcı sorusuna göre relevant chunk retrieval
7. GPT-4o ile cevap üretimi
8. Kaynakların frontend'de gösterilmesi

Bu testlerde LLM'in genel olarak tutarlı cevaplar verdiği, kaynak metinden kopmadan açıklama yapabildiği ve retrieval sonuçlarını anlamlı şekilde kullandığı gözlemlendi.

---

## 5. Çok Dilli Soru-Cevap Testleri

Sistemin çok dilli çalışıp çalışmadığını anlamak için farklı dil kombinasyonları denendi.

Örnek senaryolar:

- Türkçe belge yüklendi, İngilizce soru soruldu.
- İngilizce belge yüklendi, Türkçe soru soruldu.
- Türkçe belge hakkında Türkçe takip soruları soruldu.
- İngilizce belge hakkında İngilizce açıklama istendi.

Gözlem:

- Embedding modeli multilingual olduğu için Türkçe ve İngilizce sorularla doğru chunk'lar bulunabildi.
- GPT-4o, sorulan dilde cevap verme konusunda tutarlı davrandı.
- Türkçe belgeye İngilizce soru sorulduğunda da belge içeriği anlaşılabildi.
- İngilizce belgeye Türkçe soru sorulduğunda da doğru cevaplar üretildi.

Bu testler sonucunda sistemin temel çok dilli RAG senaryolarını desteklediği gözlemlendi.

---

## 6. Çoklu Dosya Testi

Birden fazla dosya aynı oturumda yüklenerek test edildi.

Gözlem:

- Her dosyanın chunk'ları aynı ChromaDB collection içinde saklandı.
- Chunk metadata'sında dosya adı (`source`) tutulduğu için kaynak ayrımı yapılabildi.
- Soru ilgili dosyaya ait içerikle eşleştiğinde doğru chunk'lar getirildi.
- Frontend'de yüklenen dosyalar listelendi.
- Dosya silindiğinde ilgili chunk'ların ChromaDB'den temizlendiği doğrulandı.

Bu test, sistemin tek dosyaya bağlı kalmadan birden fazla belgeyle çalışabildiğini gösterdi.

---

## 7. Conversation History ve Summary Testi

Akıcı sohbet deneyimini test etmek için uzun konuşmalar yapıldı.

Test edilen durumlar:

- Önceki cevaba referans veren takip soruları
- "Bunu daha basit açıkla" gibi bağlama bağlı sorular
- Uzun konuşmadan sonra önceki konuların özetini isteme
- 10 mesajdan fazla konuşmada summary mekanizmasının devreye girmesi

Gözlem:

- Son mesajlar LLM'e gönderildiği için kısa vadeli bağlam korundu.
- Konuşma uzadığında eski mesajlar özetlenerek context'e eklendi.
- Kullanıcı geçmiş konuşmanın özetiyle ilgili soru sorduğunda sistem genel konuşma akışını hatırlayabildi.
- Bu sayede chatbot daha doğal ve akıcı bir sohbet deneyimi sundu.

---

## 8. LLM Cevap Tutarlılığı

LLM cevapları genel olarak şu açılardan değerlendirildi:

- Belgede bulunan bilgiye dayanıyor mu?
- Sorunun diline uygun cevap veriyor mu?
- Kaynak metinle çelişiyor mu?
- Belgede olmayan bilgi sorulduğunda makul davranıyor mu?
- Takip sorularında önceki bağlamı dikkate alıyor mu?

Gözlem:

- GPT-4o genel olarak tutarlı ve anlaşılır cevaplar verdi.
- RAG context'i bulunduğunda belgeye dayalı cevaplar üretildi.
- Relevant chunk bulunamadığında sistem serbest sohbet moduna düşebildi.
- Conversation history eklendikten sonra takip sorularındaki tutarlılık arttı.

---

## 9. Bilinen Sınırlamalar

Testler sonucunda bazı sınırlamalar gözlemlendi:

1. **El yazısı performansı:** Basılı metinlerde OCR başarısı yüksekken el yazılarında hata oranı artıyor.
2. **Çok yoğun ekran görüntüleri:** Metin çok sıkışık veya sayfa düzeni karmaşıksa OCR çıktısında satır sırası ve karakter hataları oluşabiliyor.
3. **Tablolu belgeler:** Metin çıkarılsa bile tablo yapısı birebir korunmuyor.
4. **PDF türüne bağlı farklar:** Dijital PDF'lerde native extraction çok başarılı; taranmış PDF'lerde kalite OCR başarısına bağlı.
5. **LLM değerlendirmesi manuel:** Cevap doğruluğu otomatik metriklerle değil manuel gözlemle değerlendirildi.

---

## 10. Genel Sonuç

Testler sonucunda sistemin temel hedeflerini karşıladığı görüldü:

- PDF ve görsel dosyalardan metin çıkarılabiliyor.
- Türkçe ve İngilizce basılı metinlerde OCR başarısı yüksek.
- El yazısı gibi zor koşullarda başarım düşse de sistemin sınırları anlaşılabiliyor.
- Çıkarılan metinler chunk'lara ayrılıp embedding'e dönüştürülüyor.
- ChromaDB üzerinden ilgili parçalar bulunabiliyor.
- GPT-4o ile belgeye dayalı ve çok dilli cevaplar üretilebiliyor.
- Birden fazla dosya aynı oturumda destekleniyor.
- Conversation summary sayesinde uzun sohbetlerde bağlam kaybı azaltılıyor.

Bu nedenle proje, belge tabanlı çok dilli RAG chatbot hedefi için işlevsel ve testlerle doğrulanmış bir prototip olarak değerlendirildi.
