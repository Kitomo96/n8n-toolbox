# =================================================================================
# FINAL AND COMPLETE main.py - Version 3.2.0
# This is the full code, including all endpoints and security features.
# =================================================================================
import os
import uuid
import subprocess
import json
import logging
import time
import mimetypes
import secrets
from typing import Optional

# --- Imports for FastAPI and Security ---
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# --- Imports for Tools ---
import httpx
import cv2
import numpy as np
from PIL import Image
import pytesseract
from exiftool import ExifToolHelper

# NOTE: original had: from crawl4ai import crawl
# Patch: use modern crawl4ai API with markdown generator (avoids nulls)
try:
    from crawl4ai import AsyncWebCrawler as _Crawler, CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
except Exception:
    _Crawler = None
    CrawlerRunConfig = None
    CacheMode = None
    DefaultMarkdownGenerator = None

# --- Helper for local development with .env file ---
from dotenv import load_dotenv
load_dotenv()

# --- Configuration from Environment Variables ---
APP_VERSION = os.getenv("APP_VERSION", "3.2.0-dev")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Logging Configuration ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Initialization & Response Models ---
app = FastAPI(
    title="n8n External Toolbox API",
    description="A resilient, observable, and production-ready API for external tools.",
    version=APP_VERSION
)

class SuccessResponse(BaseModel):
    ok: bool = True
    data: dict

class ErrorResponse(BaseModel):
    ok: bool = False
    error: dict

# --- Security Dependency for API Key Verification ---
API_KEY_NAME = "X-API-Key"
SECRET_KEY = os.getenv("API_KEY")

api_key_header_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header_scheme)):
    """Verifies that the API key provided in the 'X-API-Key' header is valid."""
    if not SECRET_KEY:
        logging.error("API_KEY environment variable is not set on the server.")
        raise HTTPException(status_code=500, detail={"ok": False, "error": {"message": "API Key is not configured on the server."}})
    
    if not api_key:
        raise HTTPException(status_code=403, detail={"ok": False, "error": {"message": "An API Key is required in the 'X-API-Key' header."}})

    if not secrets.compare_digest(api_key, SECRET_KEY):
        raise HTTPException(status_code=403, detail={"ok": False, "error": {"message": "Invalid API Key."}})

# --- Structured Logging Middleware ---
@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    logging.info(f"req_id={request_id} start method={request.method} path={request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logging.info(f"req_id={request_id} done status_code={response.status_code} process_time_ms={process_time:.2f}")
    return response

# --- Robust File Handling Helper ---
async def get_temp_file_from_request(background_tasks: BackgroundTasks, file: Optional[UploadFile] = None, url: Optional[str] = None) -> str:
    if not file and not url:
        raise HTTPException(status_code=400, detail={"ok": False, "error": {"message": "Please provide either a file or a URL."}})

    temp_path = f"/tmp/{uuid.uuid4()}"
    
    if file:
        size = 0
        try:
            with open(temp_path, "wb") as f:
                while chunk := await file.read(CHUNK_SIZE_BYTES):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE_BYTES:
                        raise HTTPException(status_code=413, detail={"ok": False, "error": {"message": f"File size exceeds limit of {MAX_FILE_SIZE_MB} MB."}})
                    f.write(chunk)
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            raise e
    elif url:
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type")
                    extension = mimetypes.guess_extension(content_type) if content_type else os.path.splitext(url)[1]
                    if extension: temp_path += extension
                    size = 0
                    with open(temp_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > MAX_FILE_SIZE_BYTES:
                                raise HTTPException(status_code=413, detail={"ok": False, "error": {"message": f"Downloaded file size exceeds limit of {MAX_FILE_SIZE_MB} MB."}})
                            f.write(chunk)
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail={"ok": False, "error": {"message": f"Error downloading file: {e}"}})

    background_tasks.add_task(os.remove, temp_path)
    return temp_path

# --- Ergonomic & Health Endpoints (Public) ---
@app.get("/health", tags=["General"], response_model=SuccessResponse)
async def health_check():
    return {"ok": True, "data": {"status": "healthy"}}

@app.get("/version", tags=["General"], response_model=SuccessResponse)
async def get_version():
    return {"ok": True, "data": {"version": APP_VERSION}}

# --- Tool Endpoints (All Protected by API Key) ---

@app.post("/ocr", tags=["Tesseract"], response_model=SuccessResponse)
async def run_ocr(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    lang: str = Form("eng"), psm: int = Form(6), oem: int = Form(3),
    file: Optional[UploadFile] = None, url: Optional[str] = Form(None)
):
    temp_path = await get_temp_file_from_request(background_tasks, file, url)
    try:
        img_cv = cv2.imdecode(np.fromfile(temp_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None: raise ValueError("File is not a valid image or is corrupted.")
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = f'--psm {psm} --oem {oem}'
        text = pytesseract.image_to_string(threshold_img, lang=lang, config=config)
        return {"ok": True, "data": {"text": text, "lang": lang}}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"ok": False, "error": {"message": str(e)}})

@app.post("/exif", tags=["ExifTool"], response_model=SuccessResponse)
async def get_exif_data(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    file: Optional[UploadFile] = None, url: Optional[str] = Form(None)
):
    temp_path = await get_temp_file_from_request(background_tasks, file, url)
    try:
        with ExifToolHelper() as et:
            metadata = et.get_metadata(temp_path)
        return {"ok": True, "data": {"metadata": metadata}}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"ok": False, "error": {"message": str(e)}})

@app.post("/image/convert", tags=["Pillow"])
async def convert_image(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    format: str = Form(..., enum=["jpeg", "png", "webp"]),
    file: Optional[UploadFile] = None, url: Optional[str] = Form(None)
):
    input_path = await get_temp_file_from_request(background_tasks, file, url)
    output_path = f"{input_path}.{format}"
    background_tasks.add_task(os.remove, output_path)
    try:
        with Image.open(input_path) as img:
            if format.lower() == 'jpeg' and img.mode in ('RGBA', 'P'): img = img.convert('RGB')
            img.save(output_path, format=format)
        mime_type = f"image/{format}"
        return FileResponse(path=output_path, media_type=mime_type, filename=f"converted.{format}")
    except Exception as e:
        raise HTTPException(status_code=500, detail={"ok": False, "error": {"message": str(e)}})

@app.post("/image/crop", tags=["Pillow"])
async def crop_image(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...),
    file: Optional[UploadFile] = None, url: Optional[str] = Form(None)
):
    if w <= 0 or h <= 0 or w > 10000 or h > 10000:
        raise HTTPException(status_code=422, detail={"ok": False, "error": {"message": "Invalid dimensions. Width and height must be positive and less than 10,000 pixels."}})
    input_path = await get_temp_file_from_request(background_tasks, file, url)
    output_path = f"{os.path.splitext(input_path)[0]}_cropped.png"
    background_tasks.add_task(os.remove, output_path)
    try:
        with Image.open(input_path) as img:
            if x + w > img.width or y + h > img.height:
                 raise ValueError("Crop box is outside the image bounds.")
            cropped_img = img.crop((x, y, x + w, y + h))
            cropped_img.save(output_path, "PNG")
        return FileResponse(path=output_path, media_type="image/png", filename="cropped.png")
    except Exception as e:
        raise HTTPException(status_code=422 if isinstance(e, ValueError) else 500, detail={"ok": False, "error": {"message": str(e)}})

@app.post("/ffmpeg/convert", tags=["FFmpeg"])
async def convert_media(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    target_format: str = Form(..., enum=["mp3", "mp4", "wav", "webm"]),
    start: Optional[int] = Form(None), duration: Optional[int] = Form(None),
    file: Optional[UploadFile] = None, url: Optional[str] = Form(None)
):
    input_path = await get_temp_file_from_request(background_tasks, file, url)
    output_path = f"{os.path.splitext(input_path)[0]}.{target_format}"
    background_tasks.add_task(os.remove, output_path)
    try:
        command = ["ffmpeg", "-y"]
        if start is not None: command.extend(["-ss", str(start)])
        command.extend(["-i", input_path])
        if duration is not None: command.extend(["-t", str(duration)])
        if target_format == "mp3": command.extend(["-q:a", "0"])
        command.append(output_path)
        subprocess.run(command, check=True, capture_output=True, text=True)
        mime_type = mimetypes.guess_type(output_path)[0] or "application/octet-stream"
        return FileResponse(path=output_path, media_type=mime_type, filename=f"converted.{target_format}")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={"ok": False, "error": {"message": "FFmpeg conversion failed.", "details": e.stderr}})

# --- Crawl4AI (patched to modern API + normalized markdown) ---
@app.post("/crawl", tags=["Crawl4AI"], response_model=SuccessResponse)
async def run_crawl(
    api_key: str = Depends(verify_api_key),
    url: str = Form(...)
):
    if not url:
        raise HTTPException(status_code=400, detail={"ok": False, "error": {"message": "URL parameter is required."}})
    try:
        if _Crawler is None:
            raise RuntimeError("crawl4ai.AsyncWebCrawler is not available in this package version")

        run_cfg = None
        if CrawlerRunConfig and DefaultMarkdownGenerator and CacheMode:
            run_cfg = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=DefaultMarkdownGenerator()
            )

        async with _Crawler() as crawler:
            res = await crawler.arun(url=url, config=run_cfg) if run_cfg else await crawler.arun(url=url)

        # --- NEW, MORE ROBUST EXTRACTION LOGIC ---
        md = None
        
        # Sequentially check for the most likely attributes that contain the content
        possible_attributes = ['markdown', 'text', 'content', 'cleaned_html']
        for attr in possible_attributes:
            content = getattr(res, attr, None)
            if content and isinstance(content, str) and content.strip():
                md = content
                break  # Stop as soon as we find valid content

        # If markdown was an object, try to extract from its properties (for older versions)
        if not md:
            mark_obj = getattr(res, 'markdown', None)
            if hasattr(mark_obj, '__dict__'): # Check if it's an object with attributes
                 md = getattr(mark_obj, "fit_markdown", None) or \
                      getattr(mark_obj, "raw_markdown", None) or \
                      getattr(mark_obj, "markdown_with_citations", None)

        # Final fallback to an empty string if nothing was found
        final_markdown = md or ""

        return {"ok": True, "data": {"markdown_content": final_markdown, "source_url": url}}
    except Exception as e:
        logging.error(f"An error occurred during crawling for {url}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={"ok": False, "error": {"message": f"An error occurred during crawling: {str(e)}"}})
