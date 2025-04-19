import logging

# Configure logging to suppress less critical logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# Import the configuration file
from .config import MAX_SEED_VALUE, DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STRENGTH

class StableDiffusionGenerator:
    def __init__(self, model_id=None, use_gpu=True):
        """
        Initialize the Stable Diffusion generator (local model preferred, download if missing).

        Args:
        model_id (str): The local path to the model directory or HuggingFace model id. If None, uses default local path.
        use_gpu (bool): Whether to use GPU for inference.
        """
        # Define the local models directory and default model path
        models_dir = os.path.join(os.path.dirname(__file__), "../../models/")
        default_hf_id = "stabilityai/stable-diffusion-2-1-base"
        default_local_model_path = os.path.join(models_dir, "stabilityai-stable-diffusion-2-1-base")
        local_model_path = model_id if model_id is not None else default_local_model_path
        hf_id = model_id if model_id is not None and ("/" in model_id) else default_hf_id

        # If local model not found, download from HuggingFace
        if not os.path.exists(local_model_path):
            print(f"Local model not found at {local_model_path}. Downloading from HuggingFace ({hf_id})...")
            self.pipeline = StableDiffusionPipeline.from_pretrained(hf_id, cache_dir=models_dir)
            self.pipeline.save_pretrained(local_model_path)
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(local_model_path, local_files_only=True)

        # Set the device based on GPU (CUDA), Apple Silicon (MPS), or CPU
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.float16
        elif use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            torch_dtype = torch.float16
            print("Using Apple Silicon MPS acceleration.")
        else:
            self.device = "cpu"
            torch_dtype = torch.float32
            if use_gpu:
                print("Warning: No GPU (CUDA or MPS) found. Running on CPU, which will be slow.")

        self.pipeline.to(self.device, dtype=torch_dtype)

        # Ensure self.pipe is properly initialized
        self.pipe = self.pipeline

        # Use more efficient scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(self.device)
        print("Model loaded successfully!")

        # Initialize img2img pipeline from local only (download if missing)
        if not os.path.exists(local_model_path):
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(hf_id, cache_dir=models_dir, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
            self.img2img_pipe.save_pretrained(local_model_path)
        else:
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=True
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
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        width=DEFAULT_WIDTH, 
        height=DEFAULT_HEIGHT,
        seed=None,
        init_image=None,
        strength=DEFAULT_STRENGTH  # Default value for strength
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

        # Remove forced overwriting of guidance_scale and strength so UI/inputs are respected
        # guidance_scale = 3.0  # Lower value to reduce prompt influence
        # strength = 0.4  # Lower value to preserve more of the original image

        # Generate the image
        if init_image is not None:
            result = self.img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image=init_image,
                strength=strength,  # Use the value passed from UI
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