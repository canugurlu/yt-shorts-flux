"""
Runpod Serverless Handler for FLUX.1-dev
YouTube Shorts T2I Generator
"""

import os
import base64
import io
import torch
from diffusers import FluxPipeline
from runpod.serverless.utils import rp_upload_file
from runpod.serverless.utils import rp_download_model

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

# Global pipeline variable (loaded once)
pipeline = None


def load_model():
    """Load FLUX.1-dev pipeline (called once on cold start)"""
    global pipeline

    print(f"Loading model: {MODEL_ID}")

    pipeline = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
    ).to(DEVICE)

    # Enable memory optimizations
    pipeline.enable_attention_slicing()

    # Enable xformers if available
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    print("Model loaded successfully")


def decode_base64_image(base64_string):
    """Decode base64 string to bytes"""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    return base64.b64decode(base64_string)


def encode_image_to_base64(image_bytes):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode("utf-8")


def generate_images(prompt, num_images=1, width=832, height=1536,
                    guidance_scale=3.5, num_inference_steps=28,
                    seed=None):
    """Generate images using FLUX.1-dev"""
    global pipeline

    if pipeline is None:
        load_model()

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Generate images
    print(f"Generating {num_images} images with prompt: {prompt[:100]}...")

    output = pipeline(
        prompt=[prompt] * num_images,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    images = output.images

    # Convert to base64
    results = []
    for i, img in enumerate(images):
        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        results.append({
            "index": i,
            "image_base64": encode_image_to_base64(img_bytes),
            "width": width,
            "height": height,
        })

    return results


def handler(event):
    """
    Runpod serverless handler

    Expected input:
    {
        "prompt": "A cinematic vertical shot of...",
        "num_images": 5,
        "width": 832,
        "height": 1536,
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
        "seed": 42
    }
    """
    # Load model if not loaded
    if pipeline is None:
        load_model()

    # Parse input
    input_data = event.get("input", {})

    prompt = input_data.get("prompt")
    if not prompt:
        return {
            "error": "Missing required parameter: prompt"
        }

    num_images = input_data.get("num_images", 5)
    width = input_data.get("width", 832)
    height = input_data.get("height", 1536)
    guidance_scale = input_data.get("guidance_scale", 3.5)
    num_inference_steps = input_data.get("num_inference_steps", 28)
    seed = input_data.get("seed", None)

    # Validate dimensions (must be divisible by 16 for FLUX)
    if width % 16 != 0 or height % 16 != 0:
        return {
            "error": f"Dimensions must be divisible by 16. Got: {width}x{height}"
        }

    # Generate images
    try:
        results = generate_images(
            prompt=prompt,
            num_images=num_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        return {
            "status": "success",
            "prompt": prompt,
            "model": MODEL_ID,
            "images": results,
            "count": len(results),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        "input": {
            "prompt": "A cinematic vertical shot of a futuristic city at sunset, cyberpunk style, neon lights, highly detailed, 8k",
            "num_images": 2,
            "width": 832,
            "height": 1536,
        }
    }

    result = handler(test_event)
    print(f"Generated {result.get('count', 0)} images")
