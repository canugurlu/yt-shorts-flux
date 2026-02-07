"""
Runpod Serverless Handler for FLUX.1-dev
YouTube Shorts T2I Generator
"""

import os
import base64
import io
import gc
import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

# Global pipeline variable (loaded once)
pipeline = None


def load_model():
    """Load FLUX.1-dev pipeline with memory optimizations"""
    global pipeline

    print(f"Loading model: {MODEL_ID}")

    # Memory optimization settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load with reduced memory footprint
    pipeline = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
    )

    # Move to CPU first to avoid OOM during load
    # Then enable optimizations before moving to GPU

    # Enable CPU offloading for transformer components
    pipeline.enable_model_cpu_offload()

    # Enable sequential CPU offload (more aggressive)
    # pipeline.enable_sequential_cpu_offload()

    # Enable attention slicing (reduces memory at cost of speed)
    pipeline.enable_attention_slicing()

    # Enable vae slicing (process VAE in chunks)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    print("Model loaded successfully with memory optimizations")


def clear_cache():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


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

    # Clear cache before generation
    clear_cache()

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Generate images ONE AT A TIME to save memory
    results = []

    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}...")

        output = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        img = output.images[0]

        # Convert to base64
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        results.append({
            "index": i,
            "image_base64": encode_image_to_base64(img_bytes),
            "width": width,
            "height": height,
        })

        # Clear cache after each image
        del output
        del img
        clear_cache()

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

    # Limit num_images to prevent OOM
    if num_images > 5:
        num_images = 5

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
            "num_images": 1,
            "width": 832,
            "height": 1536,
        }
    }

    result = handler(test_event)
    print(f"Generated {result.get('count', 0)} images")
