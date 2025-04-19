import os
import sys
import shutil
import warnings

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

# Clear the output folder at app launch
output_dir = os.path.join(os.path.dirname(__file__), '../../output')
if os.path.exists(output_dir):
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.warning(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == "__main__":
    import uvicorn
    logging.info(f"Starting server at {host}:{port} with reload enabled")
    uvicorn.run(
        "src.ui.app:app",
        host=host,
        port=port,
        reload=True,
        factory=False
    )
