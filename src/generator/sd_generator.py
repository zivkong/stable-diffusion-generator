import logging

# Suppress debug and info logs from specific libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image
import os

class StableDiffusionGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", use_gpu=True):
        """
        Initialize Stable Diffusion generator with specified model
        
        Args:
            model_id (str): HuggingFace model ID
            use_gpu (bool): Whether to use GPU for inference
        """
        self.model_id = model_id
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_id} on {self.device}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Use more efficient scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        print("Model loaded successfully!")
        
        # Initialize img2img pipeline
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.img2img_pipe = self.img2img_pipe.to(self.device)

        print("Img2Img pipeline loaded successfully!")

        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Disable safety checker to allow NSFW images
        self.pipe.safety_checker = None
        self.img2img_pipe.safety_checker = None
    
    def generate_image(
        self, 
        prompt, 
        negative_prompt="", 
        num_inference_steps=30,
        guidance_scale=7.5,
        width=512, 
        height=512,
        seed=None,
        init_image=None,
        strength=0.75  # Default value for strength
    ):
        """
        Generate an image based on the provided prompt
        
        Args:
            prompt (str): Positive prompt describing what should be in the image
            negative_prompt (str): Things to avoid in the image
            num_inference_steps (int): Number of denoising steps (more = better quality, slower)
            guidance_scale (float): How closely to follow the prompt (higher = more prompt adherence)
            width (int): Image width
            height (int): Image height
            seed (int): Random seed for reproducibility
            init_image (PIL.Image or None): Initial image to guide generation
            strength (float): Strength of the transformation when using an initial image
            
        Returns:
            PIL.Image: Generated image
        """
        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Ensure init_image is converted to a PIL.Image if not already
        if init_image is not None and not isinstance(init_image, Image.Image):
            init_image = Image.open(init_image)
            init_image = init_image.convert("RGB")  # Convert to RGB mode to ensure compatibility

        # Use the strength parameter in the image generation process (if applicable)
        # This is a placeholder for actual implementation logic
        if init_image is not None:
            # Example: Adjust the strength parameter in the pipeline
            pass

        # Resize the init_image to match the specified dimensions without distortion
        if init_image is not None:
            original_width, original_height = init_image.size
            aspect_ratio = original_width / original_height

            if aspect_ratio > width / height:
                new_width = width
                new_height = int(width / aspect_ratio)
            else:
                new_height = height
                new_width = int(height * aspect_ratio)

            init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create a new image with the target dimensions and paste the resized image onto it
            padded_image = Image.new("RGB", (width, height), (0, 0, 0))  # Black padding
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            padded_image.paste(init_image, (paste_x, paste_y))
            init_image = padded_image

        # Ensure minimal changes to the original image by adjusting parameters
        guidance_scale = 3.0  # Lower value to reduce prompt influence
        strength = 0.4  # Lower value to preserve more of the original image

        # Generate the image
        if init_image is not None:
            result = self.img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image=init_image,
                strength=strength,  # Use the adjusted strength parameter
                generator=generator
            )
        else:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )

        return result.images[0]
    
    def save_image(self, image, filename=None):
        """
        Save the generated image
        
        Args:
            image (PIL.Image): Image to save
            filename (str): Optional filename, default is timestamp
            
        Returns:
            str: Path to saved image
        """
        if filename is None:
            import time
            filename = f"sd_image_{int(time.time())}.png"
            
        if not filename.endswith(".png"):
            filename += ".png"
            
        path = os.path.join("output", filename)
        image.save(path)
        print(f"Image saved to {path}")
        
        return path


# Simple test if run directly
if __name__ == "__main__":
    generator = StableDiffusionGenerator()
    
    prompt = "A beautiful sunset over a calm ocean, detailed, vibrant colors"
    negative_prompt = "blurry, distorted, low quality"
    
    image = generator.generate_image(prompt, negative_prompt)
    generator.save_image(image)