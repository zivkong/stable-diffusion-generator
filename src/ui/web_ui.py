import sys
import os
# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import torch

# Update logging to write to a file for persistent debugging
log_file_path = os.path.join(os.path.dirname(__file__), '../../output/logs.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logging.debug("Logging setup updated to write to file.")

import gradio as gr
from PIL import Image
import numpy as np
import io

# Import the configuration file
from generator.config import MAX_SEED_VALUE, DEFAULT_WIDTH, DEFAULT_HEIGHT

# Determine the best available device for computation
def get_device():
    if torch.backends.mps.is_available():
        logging.debug("MPS (Metal Performance Shaders) is available and will be used for GPU acceleration.")
        return torch.device("mps")
    else:
        logging.debug("MPS not available. Falling back to CPU.")
        return torch.device("cpu")

# Update the generator to use the appropriate device
def generate_image(prompt, negative_prompt, steps, guidance_scale, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, seed=None, use_random_seed=False, uploaded_image=None, strength=0.75):
    import random
    from generator.sd_generator import StableDiffusionGenerator

    device = get_device()
    use_gpu = device.type != "cpu"  # Determine if GPU should be used
    generator = StableDiffusionGenerator(use_gpu=use_gpu)  # Pass use_gpu instead of device

    if use_random_seed:
        seed = random.randint(0, MAX_SEED_VALUE)
    elif seed:
        seed = int(seed)
    else:
        seed = None

    # If an image is uploaded, use it as an initial input
    init_image = None
    if uploaded_image is not None:
        init_image = Image.open(uploaded_image).convert("RGB")
        init_image = init_image.resize((width, height))  # Resize to match generation dimensions

    # Generate the image
    image = generator.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        init_image=init_image,  # Pass processed init_image
        strength=strength  # Pass the strength parameter
    )

    # Add post-processing to enhance the generated image
    from PIL import ImageEnhance

    # Enhance the image after generation
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)  # Increase sharpness by 1.5x

    # Add debugging logs to trace the outputs of the generate_image function
    logging.debug(f"Generated image: {image}")
    logging.debug(f"Generated seed: {seed}")

    return image, str(seed) if seed else ""

# Create a Gradio interface for the image generation
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Textbox(label="Negative Prompt"),
        gr.Slider(1, 100, step=1, label="Steps"),
        gr.Slider(1.0, 20.0, step=0.1, label="Guidance Scale"),
        gr.Slider(256, 1024, step=64, label="Width"),
        gr.Slider(256, 1024, step=64, label="Height"),
        gr.Number(label="Seed"),
        gr.Checkbox(label="Use Random Seed"),
        gr.Image(type="filepath", label="Uploaded Image"),
        gr.Slider(0.0, 1.0, step=0.01, label="Strength")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Seed")
    ],
    title="Stable Diffusion Image Generator",
    description="Generate images using Stable Diffusion with customizable parameters."
)

# Add FastAPI integration for hot reload
from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from fastapi.staticfiles import StaticFiles

# Create a FastAPI app
app = FastAPI()

# Mount the Gradio interface to the FastAPI app
mount_gradio_app(app, interface, path="/")

# Serve static files (fonts, etc.)
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Serve manifest.json at the root
from fastapi.responses import FileResponse
@app.get("/manifest.json")
def manifest():
    manifest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../manifest.json'))
    return FileResponse(manifest_path, media_type="application/json")

import dotenv

# Load environment variables from .env file
dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Define default values for host and port
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Update the host and port to use environment variables
host = os.getenv("UI_ADDRESS", DEFAULT_HOST)
port = int(os.getenv("UI_PORT", DEFAULT_PORT))

# Add detailed logging for debugging
logging.debug("Loading environment variables from .env file.")
if not os.path.exists(os.path.join(os.path.dirname(__file__), '../../.env')):
    logging.error(".env file not found at the specified path.")
else:
    logging.debug(".env file found and loaded successfully.")

logging.debug(f"Host: {host}, Port: {port}")
logging.debug("Starting the FastAPI application with Uvicorn.")

# Run the app with uvicorn for hot reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.ui.web_ui:app", host=host, port=port, reload=True)