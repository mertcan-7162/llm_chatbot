# TESTING — Test ve Doğrulama

## 1. Test Stratejisi

Sistem manuel olarak farklı belge tipleri, diller ve senaryo kombinasyonlarıyla test edildi. Testler hem CLI aracı (`scripts/extract_text.py`) hem de web UI üzerinden yapıldı. Amaç: metin çıkarma doğruluğu, RAG retrieval kalitesi ve LLM cevap tutarlılığını ölçmek.

---

## 2. Test Edilen Belge Tipleri

### 2.1 Türkçe Görsel — `2.mehmet.png`

**İçerik:** Fatih Sultan Mehmed Vikipedi makalesi ekran görüntüsü.

**OCR sonucu:** PaddleOCR metni büyük ölçüde doğru çıkardı. Türkçe karakterler (ş, ç, ö, ü, ğ, ı) çoğunlukla doğru okundu ancak bazı eksiklikler var:

- "padişahıdır" → "padiahidir" (ş ve dotted-ı kayıpları)
- "şehzade" → "ehzade"
- "yaklaşık" → "yaklaik"
- "başladı" → "balad"

**Değerlendirme:** OCR, ana içeriği koruyarak okunabilir bir metin üretiyor. Türkçe özel karakterlerde kısmi kayıplar var ama cümle anlamı büyük ölçüde korunuyor. RAG pipeline'ında bu metin üzerinden sorulan sorulara doğru cevaplar üretildi.

**Örnek sorular ve cevaplar:**

| Soru | Beklenen | Sonuç |
|------|----------|-------|
| "İstanbul ne zaman fethedildi?" | 29 Mayıs 1453 | Doğru cevaplandı |
| "Fatih kaç yaşında öldü?" | 49 | Doğru cevaplandı |
| "Annesi kimdir?" | Hüma Hatun | Doğru cevaplandı |

### 2.2 Almanca Görsel — `Text_entropy.png`

**İçerik:** Rudolf Clausius'un entropi tanımını yaptığı orijinal Almanca metin.

**OCR sonucu:** Almanca metin yüksek doğrulukla çıkarıldı. Umlauts (ä, ö, ü) ve ß karakterleri doğru okundu. Eski Almanca tipografi ve ligatures sorunsuz işlendi.

**Çıkarılan metin (kısaltılmış):**
> "Sucht man für S einen bezeichnenden Namen, so könnte man, ähnlich wie von der Grösse U gesagt ist..."

**Değerlendirme:** Çok iyi. Tarihî bir matbu belge olmasına rağmen OCR doğruluğu yüksek.

### 2.3 PDF — `mert_cv.pdf`

**İçerik:** Dijital olarak oluşturulmuş CV belgesi.

**Extraction yöntemi:** Native text extraction (PyMuPDF). OCR'a gerek kalmadı çünkü PDF dijital metin katmanı içeriyor.

**Sonuç:** Metin eksiksiz ve doğru çıkarıldı. Sayfa yapısı korundu.

### 2.4 PDF — `lect_03.pdf`

**İçerik:** Ders notu PDF'i.

**Extraction yöntemi:** Hibrit. Bazı sayfalar native text, bazıları (taranmış sayfalar varsa) OCR ile işlendi.

**Sonuç:** Dijital sayfalar mükemmel çıkarıldı. Taranmış sayfalar OCR ile işlendi.

### 2.5 İngilizce Görsel — `tmp_ocr_test.png`

**İçerik:** Test amaçlı İngilizce metin görseli.

**OCR sonucu:** İngilizce metin yüksek doğrulukla çıkarıldı.

---

## 3. Sistem Davranış Testleri

### 3.1 Belgede Olmayan Bilgi Sorulduğunda

**Senaryo:** Bir belge yüklendikten sonra belgede olmayan bir soru soruldu.

**Davranış:** İki durum var:

1. **Chroma'dan sonuç döner ama alakasız:** Relevance threshold (`min_score`) devreye girer. Skor eşiğin altındaysa tüm sonuçlar filtrelenir ve sistem serbest sohbet moduna düşer. LLM, RAG context'i olmadan genel bilgisiyle cevap verir.

2. **Chroma'dan sonuç döner ve eşiğin üstünde:** RAG prompt'undaki kural devreye girer: *"If the context does not contain enough information, say so clearly."* LLM bağlamda yeterli bilgi olmadığını belirtir.

### 3.2 Hiç Belge Yüklenmeden Sohbet

**Senaryo:** Kullanıcı dosya yüklemeden soru soruyor.

**Davranış:** Collection boş olduğu için (`has_documents()` → False) sistem doğrudan `LLMClient.chat()` kullanıyor. Normal bir chatbot gibi davranıyor, kaynak göstermiyor.

### 3.3 Aynı Dosyanın Tekrar Yüklenmesi

**Senaryo:** Aynı dosya iki kez yükleniyor.

**Davranış:** Chunk ID'leri UUID olduğu için aynı dosyanın chunk'ları yeni ID'lerle ekleniyor. Bu, duplicate chunk'lara yol açabilir. Bilinen bir sınırlama; deterministik ID ile çözülebilir.

### 3.4 Dosya Silme

**Senaryo:** Yüklenen bir dosya UI'dan siliniyor.

**Davranış:** `DELETE /api/v1/documents/{source}` endpoint'i, Chroma'dan `source` metadata'sına göre ilgili tüm chunk'ları siler. Diğer dosyaların chunk'ları etkilenmez.

### 3.5 Yeni Sohbet (New Chat)

**Senaryo:** Kullanıcı "New Chat" butonuna basıyor.

**Davranış:** Aktif collection tamamen silinir (tüm chunk'lar). UI sıfırlanır. Temiz bir başlangıç sağlanır.

---

## 4. Performans Gözlemleri

| İşlem | Yaklaşık Süre | Ortam |
|-------|--------------|-------|
| PaddleOCR ilk yükleme | ~5-10 saniye | CPU |
| Embedding model ilk yükleme | ~2-3 saniye | CPU |
| Tek sayfa görsel OCR | ~1-3 saniye | CPU |
| PDF native extraction (sayfa başı) | < 0.1 saniye | CPU |
| Embedding üretme (5 chunk) | < 0.5 saniye | CPU |
| Chroma sorgu | < 0.1 saniye | CPU |
| LLM cevap (GPT-4o) | ~1-3 saniye | API |

---

## 5. Bilinen Sınırlamalar

1. **Türkçe karakter kayıpları:** PaddleOCR bazı Türkçe özel karakterleri (ş, ğ, ı) yanlış okuyabiliyor. `ocr_lang` ayarı `tr` olarak ayarlı; bu, `latin_PP-OCRv5_mobile_rec` modelini yüklüyor ve tüm Latin alfabeli dilleri (Türkçe, İngilizce, Almanca vb.) destekliyor.

2. **Tablo ve yapısal veri:** Tablolu belgeler düz metin olarak çıkarılıyor. Tablo yapısı korunmuyor. Satır/sütun ilişkileri kayboluyor.

3. **Multi-page TIFF:** Çok sayfalı TIFF dosyaları tek sayfa olarak işleniyor. Yalnızca ilk frame okunuyor.

4. **Conversation history yok:** Her sorgu bağımsız işleniyor. "Bunu biraz daha açıklar mısın?" gibi takip soruları çalışmıyor.

5. **Duplicate chunk'lar:** Aynı dosya tekrar yüklendiğinde yeni UUID ile chunk'lar oluşuyor.

6. **ImagePreprocessor kullanılmıyor:** Düşük kaliteli görseller için yazılan preprocessor (kontrast artırma, sharpening) pipeline'a entegre edilmemiş durumda.

7. **Büyük dosya timeout:** Çok sayfalı büyük PDF'lerde OCR uzun sürebilir ve HTTP request timeout'a düşebilir.

---

## 6. Güvenlik Notları

- `.env.example` dosyasında gerçek API key'e benzeyen bir değer bulunuyordu. Bu, push öncesi placeholder ile değiştirilmelidir.
- Upload endpoint'inde dosya türü kontrolü var ancak dosya boyutu limiti yok.
- Authentication mekanizması bulunmuyor; public deploy için gereklidir.
