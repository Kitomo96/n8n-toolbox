import os
import uuid
import subprocess
import json
import logging
import time
import mimetypes
from typing import Optional

import httpx
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import pytesseract
from exiftool import ExifToolHelper
from crawl4ai import crawl

# --- Configuration ---
APP_VERSION = os.getenv("APP_VERSION", "3.0.0-dev")
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB

# --- API Initialization & Response Models ---
app = FastAPI(
    title="n8n External Toolbox API (v3)",
    description="A resilient, observable, and production-ready API for external tools.",
    version=APP_VERSION
)

class SuccessResponse(BaseModel):
    ok: bool = True
    data: dict

class ErrorResponse(BaseModel):
    ok: bool = False
    error: dict

# --- NEW: Structured Logging Middleware ---
@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logging.info(f"req_id={request_id} start method={request.method} path={request.url.path}")
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logging.info(
        f"req_id={request_id} done status_code={response.status_code} process_time_ms={process_time:.2f}"
    )
    return response

# --- UPGRADED: Robust File Handling Helper ---
async def get_temp_file_from_request(background_tasks: BackgroundTasks, file: Optional[UploadFile] = None, url: Optional[str] = None) -> str:
    if not file and not url:
        raise HTTPException(status_code=400, detail={"ok": False, "error": {"message": "Please provide either a file or a URL."}})

    temp_path = f"/tmp/{uuid.uuid4()}"
    
    if file:
        # Streamed upload with size limit
        size = 0
        try:
            with open(temp_path, "wb") as f:
                while chunk := await file.read(CHUNK_SIZE_BYTES):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE_BYTES:
                        raise HTTPException(status_code=413, detail={"ok": False, "error": {"message": f"File size exceeds limit of {MAX_FILE_SIZE_MB} MB."}})
                    f.write(chunk)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
    elif url:
        # URL download with extension derivation
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
                    response.raise_for_status()
                    
                    content_type = response.headers.get("content-type")
                    extension = mimetypes.guess_extension(content_type) if content_type else os.path.splitext(url)[1]
                    if extension:
                        temp_path += extension

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

# --- Ergonomic & Health Endpoints (Unchanged) ---
@app.get("/health", tags=["General"], response_model=SuccessResponse)
async def health_check():
    return {"ok": True, "data": {"status": "healthy"}}

@app.get("/version", tags=["General"], response_model=SuccessResponse)
async def get_version():
    return {"ok": True, "data": {"version": APP_VERSION}}

# --- Tool Endpoints (Refactored to use new helper and add validation) ---

@app.post("/ocr", tags=["Tesseract"], response_model=SuccessResponse)
async def run_ocr(
    background_tasks: BackgroundTasks,
    lang: str = Form("eng"), psm: int = Form(6), oem: int = Form(3),
    file: Optional[UploadFile] = None, url: Optional[str] = Form(None)
):
    temp_path = await get_temp_file_from_request(background_tasks, file, url)
    try:
        img_cv = cv2.imdecode(np.fromfile(temp_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            raise ValueError("File is not a valid image or is corrupted.")
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        config = f'--psm {psm} --oem {oem}'
        text = pytesseract.image_to_string(threshold_img, lang=lang, config=config)
        return {"ok": True, "data": {"text": text, "lang": lang}}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"ok": False, "error": {"message": str(e)}})

@app.post("/image/crop", tags=["Pillow"])
async def crop_image(
    background_tasks: BackgroundTasks,
    x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...),
    file: Optional[UploadFile] = None, url: Optional[str] = Form(None)
):
    # UPGRADED: Input validation
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
    target_format: str = Form(..., enum=["mp3", "mp4", "wav", "webm"]),
    start: Optional[int] = Form(None), duration: Optional[int] = Form(None),
    file: Optional[UploadFile] = None, url: Optional[str] = Form(None)
):
    input_path = await get_temp_file_from_request(background_tasks, file, url)
    output_path = f"{os.path.splitext(input_path)[0]}.{target_format}"
    background_tasks.add_task(os.remove, output_path)
    
    try:
        # UPGRADED: Fast-seek by putting -ss before -i
        command = ["ffmpeg", "-y"] # -y overwrites output files
        if start is not None:
            command.extend(["-ss", str(start)])
        
        command.extend(["-i", input_path])

        if duration is not None:
            command.extend(["-t", str(duration)])
        
        if target_format == "mp3":
            command.extend(["-q:a", "0"])
        
        command.append(output_path)
        
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        mime_type = mimetypes.guess_type(output_path)[0] or "application/octet-stream"
        return FileResponse(path=output_path, media_type=mime_type, filename=f"converted.{target_format}")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={"ok": False, "error": {"message": "FFmpeg conversion failed.", "details": e.stderr}})

# (Other endpoints like /exif, /image/convert, /crawl can be included and follow the same robust pattern)