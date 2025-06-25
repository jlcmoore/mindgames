"""
Author: Cameron Jones
Date: October, 2024

General FastAPI app to run the API and serve the frontend.
"""

import os
import logging

from fastapi import FastAPI, Response, Request
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.api import app as api_app  # Import your API routes from api.py

# By default, don't show the docs. This can be overridden by the imported app.
app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

# Mount the API routes
app.include_router(api_app.router)

## API Call Logging

# Ensure the log directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Check if handlers are already set (to prevent duplicate handlers during reloads)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.handlers:
    root_logger.handlers.clear()

# Create the file handler
file_handler = logging.FileHandler("logs/main.log")
file_handler.setLevel(logging.DEBUG)

# Create the stream handler for stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Define the formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Configure the root logger with both handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

# Use the module-level logger
logger = logging.getLogger(__name__)


def log_info(req_body, res_body):
    """Log the request and response body."""
    logger.debug(f"Request Body: {req_body}")
    logger.debug(f"Response Body: {res_body}")


@app.middleware("http")
async def log_request_and_response(request: Request, call_next):
    req_body = await request.body()
    response = await call_next(request)

    res_body = b""
    async for chunk in response.body_iterator:
        res_body += chunk

    task = BackgroundTask(log_info, req_body, res_body)
    return Response(
        content=res_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=task,
    )


# Serve static files (HTML, JS, CSS)
frontend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "frontend/dist/")
)
app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(frontend_dir, "assets")),
    name="static",
)


# Serve the main frontend page for any non-API routes
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_frontend(full_path: str):
    """Serve index.html for non-API routes to let Vue Router handle routing."""
    with open(os.path.join(frontend_dir, "index.html"), encoding="utf-8") as f:
        return f.read()


# CORS middleware for development

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Vite server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
