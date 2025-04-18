import logging

# Set up logging to capture debug information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("Starting the web_ui.py script...")

import gradio as gr
from PIL import Image
import numpy as np
import io

def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height, seed, use_random_seed):
    """Generate an image using Stable Diffusion."""
    import random
    from sd_generator import StableDiffusionGenerator

    generator = StableDiffusionGenerator()

    if use_random_seed:
        seed = random.randint(0, 2147483647)
    elif seed:
        seed = int(seed)
    else:
        seed = None

    image = generator.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed
    )
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
            width = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height = gr.Slider(256, 1024, value=512, step=64, label="Height")
            seed = gr.Textbox(label="Seed (optional)", placeholder="Enter a seed for reproducibility")
            use_random_seed = gr.Checkbox(label="Use Random Seed", value=True)
            generate_button = gr.Button("Generate Image")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")
            output_seed = gr.Textbox(label="Seed Used")

    generate_button.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed, use_random_seed],
        outputs=[output_image, output_seed]
    )

if __name__ == "__main__":
    logging.debug("Launching the Gradio interface...")
    demo.launch()
    logging.debug("Gradio interface launched successfully.")