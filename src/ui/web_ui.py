import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import torch

# Set up logging to capture detailed debug information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("web_ui.py script initialized.")

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

# Launch the Gradio interface
if __name__ == "__main__":
    server_name = os.getenv("SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("SERVER_PORT", 8000))
    interface.launch(server_name=server_name, server_port=server_port)