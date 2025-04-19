import logging

# Configure logging to suppress less critical logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import torch
import os
import sys

from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

try:
    from accelerate import infer_auto_device_map, dispatch_model
    accelerate_available = True
except ImportError:
    accelerate_available = False

# Import the configuration file
from .config import DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE, DEFAULT_WIDTH, DEFAULT_HEIGHT

class StableDiffusionGenerator:
    def __init__(self, model_id=None, use_gpu=True, use_accelerate=True, use_torch_compile=True):
        """
        Initialize the Stable Diffusion generator (SDXL Lightning by default, download if missing).
        """
        models_dir = os.path.join(os.path.dirname(__file__), "../../models/")
        default_hf_id = "stabilityai/stable-diffusion-xl-base-1.0"
        default_local_model_path = os.path.join(models_dir, "stabilityai-stable-diffusion-xl-base-1.0")
        local_model_path = model_id if model_id is not None else default_local_model_path
        hf_id = model_id if model_id is not None and ("/" in model_id) else default_hf_id

        # Prefer local model, fallback to HuggingFace if missing
        if not os.path.exists(local_model_path):
            logging.info(f"Local SDXL Lightning model not found at {local_model_path}. Downloading from HuggingFace ({hf_id})...")
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(hf_id, cache_dir=models_dir, torch_dtype=torch.float16)
            self.pipeline.save_pretrained(local_model_path)
        else:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(local_model_path, local_files_only=True, torch_dtype=torch.float16)

        # Device and dtype selection
        self.device = "cpu"
        torch_dtype = torch.float32
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
                torch_dtype = torch.float16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
                torch_dtype = torch.float16
                logging.info("Using Apple Silicon MPS acceleration.")
            else:
                logging.warning("No GPU (CUDA or MPS) found. Running on CPU, which will be slow.")

        # Move pipeline to device and dtype in one call
        self.pipe = self.pipeline.to(self.device, torch_dtype)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        # --- torch.compile for further speedup (PyTorch 2.0+) ---
        python_version = sys.version_info
        torch_compile_supported = (
            use_torch_compile and hasattr(torch, "compile") and python_version.major == 3 and python_version.minor < 12
        )
        if torch_compile_supported:
            try:
                self.pipe = torch.compile(self.pipe, mode="reduce-overhead")
                logging.info(f"Pipeline compiled with torch.compile on {self.device}.")
            except Exception as e:
                logging.warning(f"torch.compile failed: {e}")
        elif use_torch_compile:
            logging.info("torch.compile is not supported on Python 3.12+. Skipping torch.compile.")

        # --- accelerate for optimized inference (optional) ---
        if use_accelerate and accelerate_available and hasattr(self.pipe, "named_parameters"):
            try:
                device_map = infer_auto_device_map(self.pipe)
                self.pipe = dispatch_model(self.pipe, device_map=device_map)
                logging.info("Pipeline optimized with accelerate.")
            except Exception as e:
                logging.warning(f"accelerate optimization failed: {e}")
        elif use_accelerate and not accelerate_available:
            logging.info("accelerate is not installed. Skipping accelerate optimization.")
        elif use_accelerate:
            logging.info("accelerate is not compatible with this pipeline. Skipping accelerate optimization.")

        logging.info("SDXL Model loaded successfully!")

        # Create output directory
        os.makedirs("output", exist_ok=True)
        # Disable safety checker to allow NSFW images
        self.pipe.safety_checker = None
        # Cache for img2img pipeline
        self.img2img_pipe = None

    def _validate_generate_params(self, prompt, negative_prompt, num_inference_steps, guidance_scale, strength, width, height, seed, init_image):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Prompt must be a non-empty string.")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError("negative_prompt must be a string.")
        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer.")
        if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0:
            raise ValueError("guidance_scale must be a non-negative number.")
        if not isinstance(strength, (int, float)) or not (0 <= strength <= 1):
            raise ValueError("strength must be a float between 0 and 1.")
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise ValueError("width and height must be positive integers.")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a non-negative integer or None.")
        if init_image is not None and not isinstance(init_image, Image.Image):
            raise ValueError("init_image must be a PIL.Image.Image instance or None.")

    def generate_image(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        strength=0.2,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        seed=None,
        init_image=None
    ):
        """
        Generate an image based on the provided prompt.
        If init_image is provided, use img2img mode (requires strength).
        Optimized for reduced branching and code duplication.
        """
        self._validate_generate_params(prompt, negative_prompt, num_inference_steps, guidance_scale, strength, width, height, seed, init_image)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        params = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )

        if init_image is not None:
            # Only resize if needed
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.LANCZOS)
            if self.img2img_pipe is None:
                self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.pipe.config._name_or_path,
                    torch_dtype=self.pipe.dtype
                ).to(self.device)
                self.img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.img2img_pipe.scheduler.config)
                self.img2img_pipe.safety_checker = None
            params.update({"image": init_image, "strength": strength})
            result = self.img2img_pipe(**params)
        else:
            result = self.pipe(**params)
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