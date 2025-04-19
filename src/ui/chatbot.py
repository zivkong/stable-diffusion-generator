import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*semaphore objects.*", category=UserWarning)

import os
import logging
import torch
import gradio as gr
import shutil
import random

from PIL import Image, ImageEnhance
from src.generator.config import MAX_SEED_VALUE, DEFAULT_WIDTH, DEFAULT_HEIGHT
from src.generator.sd_generator import StableDiffusionGenerator

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

# Determine the best available device for computation
def get_device():
    if torch.backends.mps.is_available():
        logging.debug("MPS (Metal Performance Shaders) is available and will be used for GPU acceleration.")
        return torch.device("mps")
    else:
        logging.debug("MPS not available. Falling back to CPU.")
        return torch.device("cpu")

# --- Advanced Chatbot Implementation ---

def advanced_chatbot(message, history, negative_prompt, steps, guidance_scale, strength, width, height, uploaded_image):
    device = get_device()
    use_gpu = device.type != "cpu"
    generator = StableDiffusionGenerator(use_gpu=use_gpu)

    # Use the message as the main prompt
    prompt = message

    # Always use a random seed, ignore any provided seed value
    seed = random.randint(0, MAX_SEED_VALUE)

    # Prepare init_image if uploaded
    init_image = None
    if isinstance(uploaded_image, str) and os.path.isfile(uploaded_image):
        init_image = Image.open(uploaded_image).convert("RGB")
        init_image = init_image.resize((width, height))

    # Generate the image
    image = generator.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        strength=strength,
        width=width,
        height=height,
        seed=seed,
        init_image=init_image
    )
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)

    # Save image to output directory for chat history
    image_path = os.path.join(os.path.dirname(__file__), f"../../output/chat_{random.randint(0, 10_000_000_000)}.png")
    image.save(image_path)

    # Compose response
    response = f"Here is your image for: '{prompt}'"
    # Return a dictionary for Gradio ChatInterface compatibility
    return [{"text": response, "files": [image_path]}]

# Define advanced controls as tools for the chat
advanced_tools = [
    gr.Textbox(label="Negative Prompt", value="", interactive=True),
    gr.Slider(1, 30, value=20   , step=1, label="Steps", interactive=True),
    gr.Slider(1.0, 20, value=10, step=1, label="Guidance Scale", interactive=True),
    gr.Slider(0.0, 1.0, value=0.2, step=0.01, label="Strength", interactive=True),
    gr.Slider(512, 2048, value=1024, step=128, label="Width", interactive=True),
    gr.Slider(512, 2048, value=1024, step=128, label="Height", interactive=True),
    gr.Image(type="filepath", label="Uploaded Image", interactive=True)
]

# Create the advanced Gradio ChatInterface
chatbot = gr.ChatInterface(
    fn=advanced_chatbot,
    additional_inputs=advanced_tools,
    additional_inputs_accordion=(gr.Accordion(label="Advanced Control", open=False)),
    examples=[
        ["A futuristic cityscape at sunset", "", 10, 7.5, 0.75, 1024, 1024, None],
        ["A cat riding a bicycle", "blurry, distorted", 15, 10.0, 0.7, 1024, 1024, None]
    ],
    title="Ziv Image Generator",
    description="Give instruction, get your image."
)

# Export the Gradio interface for external use
__all__ = ['chatbot']