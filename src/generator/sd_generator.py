import logging

# Configure logging to suppress less critical logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

# Optional: import accelerate for optimized inference
try:
    from accelerate import infer_auto_device_map, dispatch_model
    accelerate_available = True
except ImportError:
    accelerate_available = False

# Import the configuration file
from .config import MAX_SEED_VALUE, DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STRENGTH

class StableDiffusionGenerator:
    def __init__(self, model_id=None, use_gpu=True, use_accelerate=True, use_torch_compile=True):
        """
        Initialize the Stable Diffusion generator (local model preferred, download if missing).

        Args:
        model_id (str): The local path to the model directory or HuggingFace model id. If None, uses default local path.
        use_gpu (bool): Whether to use GPU for inference.
        use_accelerate (bool): Whether to use accelerate for optimized inference (requires accelerate installed).
        use_torch_compile (bool): Whether to use torch.compile for further speedup (PyTorch 2.0+).
        """
        # Define the local models directory and default model path
        models_dir = os.path.join(os.path.dirname(__file__), "../../models/")
        default_hf_id = "stabilityai/stable-diffusion-xl-base-1.0"
        default_local_model_path = os.path.join(models_dir, "stabilityai-stable-diffusion-xl-base-1.0")
        local_model_path = model_id if model_id is not None else default_local_model_path
        hf_id = model_id if model_id is not None and ("/" in model_id) else default_hf_id

        # If local model not found, download from HuggingFace
        if not os.path.exists(local_model_path):
            print(f"Local SDXL model not found at {local_model_path}. Downloading from HuggingFace ({hf_id})...")
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(hf_id, cache_dir=models_dir)
            self.pipeline.save_pretrained(local_model_path)
        else:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(local_model_path, local_files_only=True)

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
        self.pipe = self.pipeline
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        # --- torch.compile for further speedup (PyTorch 2.0+) ---
        import sys
        python_version = sys.version_info
        torch_compile_supported = (
            use_torch_compile and hasattr(torch, "compile") and python_version.major == 3 and python_version.minor < 12
        )
        if torch_compile_supported:
            try:
                self.pipe = torch.compile(self.pipe, mode="reduce-overhead")
                print(f"Pipeline compiled with torch.compile on {self.device}.")
            except Exception as e:
                print(f"torch.compile failed: {e}")
        elif use_torch_compile:
            print("torch.compile is not supported on Python 3.12+. Skipping torch.compile.")

        # --- accelerate for optimized inference (optional) ---
        # Only attempt accelerate if pipeline has 'named_parameters' (i.e., is a nn.Module)
        if use_accelerate and accelerate_available and hasattr(self.pipe, "named_parameters"):
            try:
                device_map = infer_auto_device_map(self.pipe)
                self.pipe = dispatch_model(self.pipe, device_map=device_map)
                print("Pipeline optimized with accelerate.")
            except Exception as e:
                print(f"accelerate optimization failed: {e}")
        elif use_accelerate and not accelerate_available:
            print("accelerate is not installed. Skipping accelerate optimization.")
        elif use_accelerate:
            print("accelerate is not compatible with this pipeline. Skipping accelerate optimization.")

        print("SDXL Model loaded successfully!")

        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Disable safety checker to allow NSFW images
        self.pipe.safety_checker = None

    def generate_image(
        self, 
        prompt, 
        negative_prompt="", 
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        width=DEFAULT_WIDTH, 
        height=DEFAULT_HEIGHT,
        seed=None,
        init_image=None,  # Ignored for text-to-image only
        strength=DEFAULT_STRENGTH  # Ignored for text-to-image only
    ):
        """
        Generate an image based on the provided prompt (text-to-image only)
        """
        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Only use text-to-image pipeline
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