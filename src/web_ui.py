import logging
import torch

# Set up logging to capture debug information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Starting the web_ui.py script...")

import gradio as gr
from PIL import Image
import numpy as np
import io

# Check if MPS (Metal Performance Shaders) is available for GPU acceleration
def get_device():
    if torch.backends.mps.is_available():
        logging.debug("Using MPS for GPU acceleration.")
        return torch.device("mps")
    else:
        logging.debug("MPS not available. Falling back to CPU.")
        return torch.device("cpu")

# Update the generator to use the appropriate device
def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height, seed, use_random_seed, uploaded_image, strength):
    import random
    from sd_generator import StableDiffusionGenerator

    device = get_device()
    use_gpu = device.type != "cpu"  # Determine if GPU should be used
    generator = StableDiffusionGenerator(use_gpu=use_gpu)  # Pass use_gpu instead of device

    if use_random_seed:
        seed = random.randint(0, 2147483647)
    elif seed:
        seed = int(seed)
    else:
        seed = None

    # If an image is uploaded, use it as an initial input
    init_image = None
    if uploaded_image is not None:
        init_image = Image.open(uploaded_image).convert("RGB")
        init_image = init_image.resize((width, height))  # Resize to match generation dimensions
        init_image = np.array(init_image)  # Convert to numpy array for processing

    # Generate the image
    image = generator.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
        init_image=Image.fromarray(init_image) if init_image is not None else None,  # Pass processed init_image
        strength=strength  # Pass the strength parameter
    )

    # Add post-processing to enhance the generated image
    from PIL import ImageEnhance

    # Enhance the image after generation
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)  # Increase sharpness by 1.5x

    return image, str(seed) if seed else ""

with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Image Generator")
    gr.Markdown("Generate images using Stable Diffusion right in your browser.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your image here")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="What to avoid in the image")
            steps = gr.Slider(10, 100, value=30, step=1, label="Steps")
            guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
            width = gr.Slider(128, 1024, value=512, step=64, label="Width")
            height = gr.Slider(128, 1024, value=512, step=64, label="Height")
            seed = gr.Textbox(label="Seed (optional)", placeholder="Enter a seed for reproducibility")
            use_random_seed = gr.Checkbox(label="Use Random Seed", value=True)
            uploaded_image = gr.File(label="Upload Initial Image (optional)", type="filepath")
            strength = gr.Slider(0.0, 1.0, value=0.4, step=0.1, label="Strength")

            generate_button = gr.Button("Generate Image")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")
            output_seed = gr.Textbox(label="Seed Used")

        generate_button.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed, use_random_seed, uploaded_image, strength],
            outputs=[output_image, output_seed]
        )

        # Allow Enter key to trigger the generate button
        prompt.submit(
            fn=generate_image,
            inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed, use_random_seed, uploaded_image, strength],
            outputs=[output_image, output_seed]
        )

if __name__ == "__main__":
    logging.debug("Launching the Gradio interface...")
    demo.launch()
    logging.debug("Gradio interface launched successfully.")