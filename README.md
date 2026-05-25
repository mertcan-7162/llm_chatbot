# OCR + RAG Chatbot

Belgelerinizi (PDF, görsel) yükleyin, içeriğini otomatik olarak çıkarsın ve doğal dil sorularıyla belgeleriniz hakkında sohbet edin. Deterministik OCR ile metin çıkarma, vektör veritabanıyla semantik arama ve LLM ile kanıtlı cevap üretme — hepsi tek bir uygulamada.

## Mimari

```
Kullanıcı ──▶ Web UI (Tailwind)
                │
                ▼
          FastAPI Backend
                │
        ┌───────┴───────┐
        ▼               ▼
   PDF Handler     Image Handler
   (PyMuPDF)       (PaddleOCR)
        │               │
        ▼               ▼
   Native Text    OCR + Postprocess
   veya OCR       (confidence filter,
   (hibrit)        reading order,
        │          paragraph grouping)
        └───────┬───────┘
                ▼
          TextChunker
     (recursive split, overlap,
      metadata prefix)
                │
                ▼
           Embedder
  (sentence-transformers, local)
                │
                ▼
          ChromaDB Store
    (cosine similarity, persist)
                │
                ▼
        LLM Client (OpenAI)
     ┌──────────┴──────────┐
     ▼                     ▼
  RAG modu              Chat modu
  (context + soru)      (serbest sohbet)
```

### Bileşenler

| Bileşen | Dosya | Açıklama |
|---------|-------|----------|
| FastAPI sunucu | `app/main.py` | HTTP endpoint'leri, dosya upload, statik dosya servisi |
| Pipeline | `app/pipeline/processor.py` | Ingest ve query akışını orkestre eder |
| PDF Handler | `app/pdf/handler.py` | PyMuPDF ile native extraction + OCR fallback |
| OCR Engine | `app/ocr/engine.py` | PaddleOCR wrapper, v3 format desteği |
| Postprocessor | `app/ocr/postprocessor.py` | Confidence filtre, okuma sırası, paragraf gruplama |
| Preprocessor | `app/ocr/preprocessor.py` | Kontrast/sharpening (henüz entegre değil) |
| Chunker | `app/rag/chunker.py` | Recursive character splitting + metadata prefix |
| Embedder | `app/rag/embedder.py` | sentence-transformers, local inference |
| Store | `app/rag/store.py` | ChromaDB wrapper, cosine similarity |
| LLM Client | `app/llm/client.py` | OpenAI API, RAG + serbest sohbet modları |
| Prompts | `app/llm/prompts.py` | Sistem prompt'ları |
| Schemas | `app/models/schemas.py` | Pydantic veri modelleri |
| Config | `app/config.py` | Pydantic-settings, .env okuma |
| Frontend | `app/static/index.html` | Tek dosya SPA, Tailwind, dark/light tema |
| CLI aracı | `scripts/extract_text.py` | Bağımsız metin çıkarma scripti |

---

## Kurulum

### Gereksinimler

- Python 3.10+
- pip
- OpenAI API key (soru-cevap için)

### 1. Repoyu klonla

```bash
git clone <repo-url>
cd llm_chatbot
```

### 2. Virtual environment oluştur ve aktifleştir

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# veya
venv\Scripts\activate           # Windows
```

### 3. Bağımlılıkları kur

```bash
pip install -r requirements.txt
```

> **Not:** İlk kurulumda PyTorch CPU sürümü ve PaddleOCR modelleri indirilecektir. Bu birkaç dakika sürebilir.

### 4. Ortam değişkenlerini ayarla

```bash
cp .env.example .env
```

`.env` dosyasını düzenlemek istersen düzenle:

```env
OPENAI_API_KEY=sk-your-actual-key-here
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHROMA_PERSIST_DIR=./chroma_db
OCR_LANG=tr
OCR_CONFIDENCE_THRESHOLD=0.7
CHUNK_SIZE=512
CHUNK_OVERLAP=64
RELEVANCE_THRESHOLD=0.3
```

### 5. Uygulamayı başlat

```bash
uvicorn app.main:app --reload
```

Tarayıcıda aç: **http://127.0.0.1:8000**

---

### CLI Aracı

Web UI olmadan sadece metin çıkarma yapmak için:

```bash
python scripts/extract_text.py
```

Konfigürasyonu `scripts/extract_text.yaml` dosyasından okur:

```yaml
input_path: ../example_input_files/tmp_ocr_test.png
output_path: outputs/test_output.txt
output_format: text        # veya "json"
show_ocr_details: false
```

---


## Konfigürasyon

Tüm ayarlar `.env` dosyası veya ortam değişkenleri üzerinden yönetilir:

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `OPENAI_API_KEY` | | OpenAI API anahtarı. Soru-cevap için zorunlu. |
| `LLM_MODEL` | `gpt-4o` | Kullanılacak OpenAI chat modeli. |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding modeli. Local'de çalışır. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB veri dizini. |
| `OCR_LANG` | `tr` | PaddleOCR dil ayarı (`tr` → Latin alfabeli tüm diller). |
| `OCR_CONFIDENCE_THRESHOLD` | `0.7` | OCR confidence filtre eşiği (0.0-1.0). |
| `CHUNK_SIZE` | `512` | Maksimum chunk boyutu (karakter). |
| `CHUNK_OVERLAP` | `64` | Chunk'lar arası örtüşme (karakter). |
| `RELEVANCE_THRESHOLD` | `0.3` | Varsayılan minimum benzerlik skoru (0.0-1.0). |

---

## Desteklenen Dosya Türleri

| Tür | Uzantılar | Extraction Yöntemi |
|-----|-----------|-------------------|
| PDF | `.pdf` | Native text (PyMuPDF) + OCR fallback |
| Görsel | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp` | PaddleOCR |

---

## Proje Yapısı

```
llm_chatbot/
├── app/
│   ├── static/
│   │   └── index.html          # Frontend SPA
│   ├── llm/
│   │   ├── client.py           # OpenAI LLM client
│   │   └── prompts.py          # Sistem prompt'ları
│   ├── models/
│   │   └── schemas.py          # Pydantic modelleri
│   ├── ocr/
│   │   ├── engine.py           # PaddleOCR wrapper
│   │   ├── postprocessor.py    # Metin düzenleme
│   │   └── preprocessor.py     # Görsel ön-işleme
│   ├── pdf/
│   │   └── handler.py          # PDF işleme
│   ├── pipeline/
│   │   └── processor.py        # Ana pipeline
│   ├── rag/
│   │   ├── chunker.py          # Metin parçalama
│   │   ├── embedder.py         # Embedding üretme
│   │   └── store.py            # ChromaDB store
│   ├── config.py               # Ayarlar
│   └── main.py                 # FastAPI app
├── scripts/
│   ├── extract_text.py         # CLI metin çıkarma
│   └── extract_text.yaml       # CLI konfigürasyonu
├── example_input_files/        # Örnek test dosyaları
├── .env.example                # Ortam değişkenleri şablonu
├── requirements.txt            # Python bağımlılıkları
├── DEVLOG.md                   # Geliştirme günlüğü
├── TESTING.md                  # Test ve doğrulama
└── README.md                   # Bu dosya
```

---

## Teknoloji Yığını

| Katman | Teknoloji | Neden? |
|--------|-----------|--------|
| OCR | PaddleOCR (PP-OCRv5) | Belge OCR'ında en yüksek doğruluk, bbox + confidence |
| PDF | PyMuPDF (fitz) | Hızlı native text extraction, sayfa render |
| Embedding | sentence-transformers (MiniLM) | Multilingual, local inference, hafif |
| Vektör DB | ChromaDB | Gömülü, persistent, cosine similarity |
| LLM | OpenAI GPT-4o | Güçlü reasoning, multilingual |
| Backend | FastAPI | Async, otomatik dokümantasyon, tip güvenliği |
| Frontend | Tailwind CSS | Utility-first, dark/light tema, responsive |
| Markdown | marked.js + DOMPurify | LLM cevap rendering, XSS koruması |
