# DEVLOG — Geliştirme Süreci Kaydı

## 1. Problem Tanımı ve Parçalama

Hedef: kullanıcıdan PDF veya görsel dosya alıp içeriğini çıkaran, bu içeriği vektör veritabanında saklayan ve doğal dil sorularına belgelerden kanıtlı cevaplar üreten bir RAG chatbot geliştirmek.

Problemi şu alt parçalara ayırdım:

1. **Metin çıkarma (extraction):** PDF ve görsellerden metni doğru şekilde elde etmek.
2. **Metin parçalama (chunking):** Uzun metinleri anlamlı, LLM context'ine sığacak parçalara bölmek.
3. **Vektör temsil (embedding):** Metin parçalarını sayısal vektörlere dönüştürmek.
4. **Saklama ve arama (store):** Vektörleri depolamak ve benzerlik aramasıyla geri getirmek.
5. **Cevap üretme (LLM):** Bulunan bağlamla birlikte LLM'e soru göndermek.
6. **Arayüz (frontend):** Kullanıcıya dosya yükleme, sohbet ve yönetim imkânı sunan web UI.

---

## 2. Temel Mimari Kararlar

### 2.1 OCR vs. Multimodal LLM — Neden Deterministik OCR?

Projenin en kritik tasarım kararı buydu. İki alternatif vardı:

**Alternatif A — Multimodal LLM (GPT-4o Vision vb.):**
Görseli doğrudan LLM'e gönderip "bu görselde ne yazıyor?" diye sormak. Hızlı prototipleme için cazip görünse de ciddi dezavantajları var:

- **Hallucination riski:** LLM görseldeki metni "yorumlayarak" üretir; olmayan kelimeler ekleyebilir, sayıları yanlış okuyabilir, satır sırasını karıştırabilir. Bu bir chatbot için kabul edilebilir olsa da, belge tabanlı soru-cevap sisteminde güvenilirliği doğrudan zedeler.
- **Tekrarlanabilirlik:** Aynı görsel için aynı prompt ile bile farklı çıktılar üretilebilir.
- **Maliyet:** Her görsel için API çağrısı yapılması, büyük PDF'lerde sayfa başına token maliyeti yaratır.
- **Kontrol eksikliği:** Hangi metnin hangi konumdan okunduğuna dair bbox bilgisi alınamaz; confidence skoru yoktur.

**Alternatif B — Deterministik OCR (seçilen yaklaşım):**
PaddleOCR gibi geleneksel bir OCR motoru kullanmak:

- **Deterministik:** Aynı girdi için her zaman aynı çıktı. Hallucination riski sıfır.
- **Confidence skoru:** Her tespit için güvenilirlik skoru üretilir; düşük güvenilirlikli tespitler filtrelenebilir.
- **Bbox bilgisi:** Her metnin görseldeki konumu bilinir; satır sıralama ve paragraf gruplama yapılabilir.
- **Offline çalışabilirlik:** LLM API'sine bağımlılık olmadan metin çıkarma yapılabilir.

Bu nedenle metin çıkarma katmanında deterministik OCR tercih edildi. LLM yalnızca son aşamada, çıkarılan metin üzerinde soru-cevap için kullanılıyor.

### 2.2 Neden PaddleOCR?

OCR motoru seçiminde Tesseract, EasyOCR ve PaddleOCR değerlendirildi:

- **Tesseract:** Olgun ve yaygın, ancak özellikle Türkçe gibi Latin-dışı karakterler içeren dillerde ve düşük kaliteli taranan belgelerde doğruluğu düşük. Ön-işleme (binarization, deskew) olmadan zayıf sonuçlar veriyor.
- **EasyOCR:** Kullanımı kolay, çok dilli desteği iyi ama büyük belgelerde yavaş ve model boyutları büyük.
- **PaddleOCR:** Baidu'nun geliştirdiği, özellikle belge OCR'ında state-of-the-art sonuçlar üreten motor. PP-OCRv5 ile hem detection hem recognition kalitesi yüksek. Çok dilli destek güçlü (Türkçe, İngilizce, Almanca dahil). CPU'da makul hızda çalışıyor. Text line orientation desteği var.

PaddleOCR tercih edildi çünkü:
1. Belge odaklı OCR'da en yüksek doğruluğu sunuyor.
2. Kutu bazlı (bbox) detection + recognition mimarisi, metnin konumsal bilgisini koruyor.
3. Confidence skoru üretiyor; bu skorla düşük kaliteli tespitler filtrelenebiliyor.
4. Aktif olarak geliştiriliyor ve PP-OCRv5 ile güncel.

### 2.3 ChromaDB ve Cosine Similarity

Vektör veritabanı olarak ChromaDB seçildi:

- **Gömülü (embedded) veritabanı:** Ayrı bir sunucu kurulumu gerektirmiyor. `PersistentClient` ile diske yazıp tekrar okuyabiliyor. Proje local-first tasarlandığı için bu büyük avantaj.
- **Python-native:** FastAPI ile aynı process içinde çalışıyor, ek bağımlılık yok.
- **HNSW index:** Approximate Nearest Neighbor araması için HNSW (Hierarchical Navigable Small World) algoritması kullanıyor. Küçük-orta ölçekli veri setlerinde hızlı.

Uzaklık metriği olarak **cosine similarity** tercih edildi:

```python
metadata={"hnsw:space": "cosine"}
```

Neden cosine:
- Embedding modelleri genellikle vektörlerin yönünü (direction) anlamlı kılar, büyüklüğünü (magnitude) değil. Cosine similarity yönleri karşılaştırır.
- `normalize_embeddings=True` ile üretilen birim vektörlerde cosine distance = 1 - dot product olur; bu da hesaplamayı hızlandırır.
- Farklı uzunluktaki metinlerin embedding büyüklükleri değişebilir; cosine bu farkı normalize eder.

Skor hesaplaması: `score = 1.0 - distance`. Cosine distance 0'a yakınsa metinler çok benzer (skor ~1.0), uzaksa alakasız (skor ~0.0).

### 2.4 Embedding Modeli — Local Çalışma Tercihi

Embedding modeli olarak `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` seçildi:

- **Local inference:** Model tamamen local'de çalışıyor. OpenAI Embeddings API gibi bir dış servise bağımlılık yok. Bu, internet bağlantısı olmadan da çalışabilmesini, API maliyeti oluşturmamasını ve gizliliği sağlıyor.
- **Multilingual:** 50+ dili destekliyor. Türkçe, İngilizce, Almanca belgeler aynı vektör uzayında temsil edilebiliyor.
- **Hafif:** ~118M parametre, MiniLM tabanlı. CPU'da hızlı inference yapıyor.
- **Class-level cache:** Model ilk yüklemede belleğe alınıyor ve tekrar kullanılıyor (`Embedder._model`).

Bir uyumsuzluk notu: Kodda `"passage: "` / `"query: "` prefix'leri kullanılıyor ki bu E5 model ailesi için tasarlanmış bir konvansiyon. Mevcut model bunu gerektirmiyor; ancak ileride E5 modeline geçiş kolaylığı için bırakıldı.

### 2.5 GPT-4o — LLM Tercihi

Cevap üretme katmanında OpenAI GPT-4o tercih edildi. Ana sebep **implementasyon kolaylığı**: OpenAI'ın Python SDK'sı (`openai` paketi) olgun, iyi dokümante edilmiş ve async desteği hazır. `AsyncOpenAI` client'ı ile FastAPI'nin async yapısına doğrudan uyum sağlıyor. Chat Completions API'si basit bir `messages` array'i ile çalışıyor; system prompt + user prompt göndermek birkaç satır kod. Ayrıca GPT-4o'nun çok dilli reasoning kalitesi yüksek; Türkçe, İngilizce, Almanca belgelerdeki bağlamdan doğru çıkarım yapabiliyor.

Alternatif olarak local LLM (Ollama, llama.cpp) değerlendirildi ancak:
- Reasoning kalitesi özellikle belge tabanlı soru-cevap senaryolarında GPT-4o'nun gerisinde kalıyor.
- CPU'da inference süresi çok uzun, GPU gereksinimi ekliyor.
- Proje zaten embedding için local model kullanıyor; LLM için de local model eklemek kaynak tüketimini ikiye katlardı.

Bu nedenle OCR ve embedding local'de, LLM reasoning API üzerinden çalışıyor. Bu hibrit yaklaşım, maliyeti sadece soru-cevap aşamasıyla sınırlı tutuyor.

### 2.6 PDF ve Image İçin Ayrı Yollar

Metin çıkarma katmanında PDF ve image dosyaları farklı pipeline'lardan geçiyor:

**PDF yolu (`PDFHandler`):**
1. PyMuPDF (fitz) ile PDF açılır.
2. Her sayfa için önce native text extraction denenir (`page.get_text()`).
3. Native metin yeterli uzunluktaysa (≥ `MIN_TEXT_LENGTH`) direkt kullanılır — OCR'a gerek yok.
4. Metin kısa veya boşsa sayfa "taranmış" kabul edilir: 300 DPI'da pixmap render edilir ve PaddleOCR'a gönderilir.

Bu hibrit yaklaşım önemli çünkü:
- Dijital PDF'lerde native extraction çok daha hızlı ve doğru.
- Taranmış PDF'lerde native text boş veya çok kısa döner; OCR devreye girer.
- Bazı PDF'ler karma: bazı sayfalar dijital, bazıları taranmış. Sayfa bazlı karar verme bunu ele alıyor.

`MIN_TEXT_LENGTH` eşiği başlangıçta 50 olarak ayarlandı, sonra 30'a düşürüldü. Sıfır yerine bir eşik kullanılmasının nedeni: taranmış sayfalarda bile bazen sayfa numarası, watermark veya bozuk gizli OCR layer birkaç karakter üretebiliyor. `> 0` kontrolü bu sayfaları yanlışlıkla "dijital" kabul edip asıl içeriği kaçırabilir.

**Image yolu:**
- Doğrudan PaddleOCR'a gönderilir.
- Tek sayfa (page=0) olarak işlenir.
- Multi-page TIFF desteği henüz yok; bu gelecekte `PIL.ImageSequence` ile eklenebilir.

---

## 3. OCR Post-processing

Ham OCR çıktısı düzensiz metin parçalarıdır. Bunu okunabilir metne dönüştürmek için `OCRPostprocessor` geliştirildi:

1. **Confidence filtreleme:** Eşik altındaki tespitler atılır (varsayılan 0.7).
2. **Okuma sırası:** Bbox konumlarına göre yukarıdan aşağıya, soldan sağa sıralama.
3. **Paragraf gruplama:** Satırlar arası dikey boşluğa göre paragraflar tespit edilir.
4. **Metin birleştirme:** Satırlar boşlukla, paragraflar çift yeni satırla birleştirilir.

PaddleOCR v3 ile output formatı değiştiği tespit edildi. Eski format `[[bbox, (text, confidence)], ...]` dönerken yeni format `{"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}` dict yapısında dönüyor. Her iki formatı da destekleyen parser yazıldı (`_is_paddleocr_v3_result` ile dinamik tespit).

---

## 4. Chunking Stratejisi

Recursive character-based chunking uygulandı. Ayraç öncelik sırası:

1. `\n\n` — paragraf sınırı
2. `\n` — satır sınırı
3. `. ` — cümle sınırı
4. `" "` — kelime sınırı
5. `""` — karakter kesimi (son çare)

Varsayılan parametreler: `chunk_size=512`, `chunk_overlap=64`.

Chunk'ların başına metadata prefix'i eklendi: `[Source: dosya.pdf | Page: 1]`. Bu, embedding'de de dosya adı ve sayfa bilgisinin yer almasını sağlıyor; kullanıcı dosya adıyla soru sorduğunda daha alakalı sonuçlar dönüyor.

---

## 5. Conversation History ve Summary Mekanizması

İlk tasarımda her sorgu bağımsız olarak LLM'e gönderiliyordu; önceki soru-cevap çiftleri hatırlanmıyordu. Bu, takip sorularının ("bunu açıkla", "peki ikinci bölüm?") çalışmamasına neden oluyordu çünkü LLM "bunu" derken neyi kastettiğini bilemiyordu.

Bu sorunu çözmek için **sliding window + summary** hibrit yaklaşımı uygulandı:

### 5.1 Prompt Yapısı

LLM'e gönderilen `messages` dizisi şu sırayla oluşturuluyor:

1. **System prompt** — RAG veya serbest sohbet moduna göre seçilir.
2. **Conversation summary** — Yalnızca sohbet 10 mesajı aştığında, eski mesajların özeti.
3. **Relevant document chunks** — ChromaDB'den getirilen belge parçaları (RAG modunda).
4. **Son 6 mesaj** — Son 3 tur (user + assistant) olduğu gibi eklenir.
5. **Mevcut kullanıcı sorusu** — Aktif sorgu.

### 5.2 Neden Sliding Window + Summary?

Tüm sohbet geçmişini olduğu gibi göndermek token bütçesini hızla tüketir. Sadece son N mesajı göndermek ise uzun konuşmalarda erken bağlamın kaybolmasına yol açar. Hibrit yaklaşım ikisinin avantajını birleştiriyor:

- **Son 6 mesaj (sliding window):** Takip soruları ve yakın bağlam referansları ("bunu", "onu", "yukarıdaki") için yeterli.
- **Summary (10+ mesajdan sonra):** Eski mesajların LLM tarafından 200 kelimelik bir özete sıkıştırılması. Konuşmanın genel akışı, tartışılan konular ve varılan sonuçlar korunuyor; token maliyeti sabit kalıyor.

### 5.3 Uygulama Detayları

- **Frontend:** `state.conversationHistory` dizisinde tüm mesajları tutuyor. Her başarılı sorgu sonrası user ve assistant mesajlarını ekliyor. "New Chat" butonunda diziyi sıfırlıyor.
- **Backend (`Pipeline.query`):** Gelen `conversation_history`'den son 6 mesajı ayırıyor. Toplam mesaj sayısı 10'u aşarsa, eski kısmı `LLMClient.summarize()` ile özetliyor.
- **`LLMClient._build_messages`:** Tüm bileşenleri (system prompt, summary, document context, recent messages, current query) doğru sırada birleştirip OpenAI API'sine gönderiyor.
- **Summary prompt:** Kısa, düşük sıcaklıkta (temperature=0) ve 300 max token ile çalışıyor; hızlı ve tutarlı özetler üretiyor.

Bu yapı sayesinde kullanıcı "bunu daha basit açıkla" veya "önceki sorumdaki belgeyi tekrar özetle" gibi takip soruları sorabiliyor ve chatbot bağlamı doğru şekilde takip edebiliyor.

