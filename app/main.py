from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.pipeline.processor import Pipeline

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initializing pipeline...")
    pipeline = Pipeline()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="OCR + RAG + LLM Pipeline",
    description="Deterministic OCR text extraction with RAG-powered LLM reasoning",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not built")
    return FileResponse(index_path)


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def _get_pipeline() -> Pipeline:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    collection_name: str = "default",
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        p = _get_pipeline()
        result = await p.ingest(tmp_path, collection_name, original_filename=file.filename)
        return result
    except Exception as e:
        logger.exception("Ingest failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    p = _get_pipeline()
    try:
        result = await p.query(
            query=request.query,
            collection_name=request.collection_name,
            n_results=request.n_results,
            min_score=request.min_score,
        )
        return result
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents")
async def list_documents(collection_name: str = "default"):
    p = _get_pipeline()
    sources = p.store.list_sources(collection_name)
    return {"documents": sources}


@app.delete("/api/v1/documents/{source:path}")
async def delete_document(source: str, collection_name: str = "default"):
    p = _get_pipeline()
    deleted = p.store.delete_by_source(collection_name, source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{source}'")
    return {"message": f"Deleted {deleted} chunks for '{source}'", "deleted": deleted}


@app.get("/api/v1/collections")
async def list_collections():
    p = _get_pipeline()
    collections = p.store.list_collections()
    return {"collections": collections}


@app.delete("/api/v1/collections/{name}")
async def delete_collection(name: str):
    p = _get_pipeline()
    try:
        p.store.delete_collection(name)
        return {"message": f"Collection '{name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
