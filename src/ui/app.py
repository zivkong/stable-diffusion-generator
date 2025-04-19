import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


import logging
import dotenv

from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from src.ui.chatbot import chatbot

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

app = FastAPI()
app = mount_gradio_app(app, chatbot, path="")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

host = os.getenv("UI_ADDRESS", DEFAULT_HOST)
port = int(os.getenv("UI_PORT", DEFAULT_PORT))

if __name__ == "__main__":
    import uvicorn
    logging.info(f"Starting server at {host}:{port}")
    uvicorn.run(app, host=host, port=port)
