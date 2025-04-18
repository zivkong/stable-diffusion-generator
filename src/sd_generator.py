import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
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
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
    
    def generate_image(
        self, 
        prompt, 
        negative_prompt="", 
        num_inference_steps=30,
        guidance_scale=7.5,
        width=512, 
        height=512,
        seed=None
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
            
        Returns:
            PIL.Image: Generated image
        """
        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        # Generate the image
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