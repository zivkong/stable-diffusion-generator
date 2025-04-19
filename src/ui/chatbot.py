import os
import logging
import torch
import gradio as gr

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

# Determine the best available device for computation
def get_device():
    if torch.backends.mps.is_available():
        logging.debug("MPS (Metal Performance Shaders) is available and will be used for GPU acceleration.")
        return torch.device("mps")
    else:
        logging.debug("MPS not available. Falling back to CPU.")
        return torch.device("cpu")

# --- Advanced Chatbot Implementation ---

def advanced_chatbot(message, history, negative_prompt, steps, guidance_scale, width, height, seed, use_random_seed, uploaded_image, strength):
    """
    Chatbot function for Gradio's ChatInterface. Accepts a message and advanced parameters.
    """
    import random
    device = get_device()
    use_gpu = device.type != "cpu"
    generator = StableDiffusionGenerator(use_gpu=use_gpu)

    # Use the message as the main prompt
    prompt = message

    # Handle random seed
    if use_random_seed:
        seed = random.randint(0, MAX_SEED_VALUE)
    elif seed:
        seed = int(seed)
    else:
        seed = None

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
        width=width,
        height=height,
        seed=seed,
        init_image=init_image,
        strength=strength
    )
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)

    # Save image to output directory for chat history
    image_path = os.path.join(os.path.dirname(__file__), f"../../output/chat_{random.randint(0, 10_000_000_000)}.png")
    image.save(image_path)

    # Compose response
    response = f"Here is your image for: '{prompt}'\nSeed: {seed if seed is not None else 'N/A'}"
    # Return a dictionary for Gradio ChatInterface compatibility
    return [{"text": response, "files": [image_path]}]

# Define advanced controls as tools for the chat
advanced_tools = [
    gr.Textbox(label="Negative Prompt", value="", interactive=True),
    gr.Slider(1, 100, value=30, step=1, label="Steps", interactive=True),
    gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="Guidance Scale", interactive=True),
    gr.Slider(256, 1024, value=512, step=64, label="Width", interactive=True),
    gr.Slider(256, 1024, value=512, step=64, label="Height", interactive=True),
    gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Strength", interactive=True),
    gr.Number(label="Seed", value=None, interactive=True),
    gr.Checkbox(label="Use Random Seed", value=False, interactive=True),
    gr.Image(type="filepath", label="Uploaded Image", interactive=True)
]

# Create the advanced Gradio ChatInterface
chatbot = gr.ChatInterface(
    fn=advanced_chatbot,
    additional_inputs=advanced_tools,
    additional_inputs_accordion=(gr.Accordion(label="Advanced Control", open=False)),
    examples=[
        ["A futuristic cityscape at sunset", "", 10, 7.5, 512, 512, 0.75, None, True, None],
        ["A cat riding a bicycle", "blurry, distorted", 40, 10.0, 512, 512, 0.7, 12345, False, None]
    ],
    title="Stable Diffusion Chatbot",
    description="Chat with the Stable Diffusion bot! Type your prompt and optionally adjust advanced parameters using the controls below."
)

# Export the Gradio interface for external use
__all__ = ['chatbot']