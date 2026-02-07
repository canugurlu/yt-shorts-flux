"""
Runpod Serverless Handler for FLUX.1-dev
YouTube Shorts T2I Generator
Quality Mode: Single image, maximum quality, no batch processing
"""

import os
import base64
import io
import gc
import torch
import runpod
from diffusers import FluxPipeline

# Model configuration
MODEL_ID = os.environ.get("MODEL_ID", "black-forest-labs/FLUX.1-dev")
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Global pipeline variable (loaded once)
pipeline = None


def load_model():
    """Load FLUX.1-dev pipeline - quality first, no CPU offload"""
    global pipeline

    print(f"Loading model: {MODEL_ID}")

    # Memory optimization for single image generation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load model directly to GPU (no CPU offload - preserves quality and speed)
    pipeline = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
    ).to(DEVICE)

    # VAE slicing only - doesn't affect quality, just processes in chunks
    pipeline.vae.enable_slicing()

    print("Model loaded successfully")


def clear_cache():
    """Clear GPU cache"""
    gc.collect()
    torch.cuda.empty_cache()


def encode_image_to_base64(image_bytes):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode("utf-8")


def generate_image(prompt, width=832, height=1536,
                   guidance_scale=3.5, num_inference_steps=28,
                   seed=None):
    """Generate SINGLE image using FLUX.1-dev - maximum quality"""
    global pipeline

    if pipeline is None:
        load_model()

    clear_cache()

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    print(f"Generating image: {prompt[:80]}...")

    # Single image generation - full quality
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

    clear_cache()

    return {
        "image_base64": encode_image_to_base64(img_bytes),
        "width": width,
        "height": height,
    }


def handler(job):
    """
    Runpod serverless handler - SINGLE IMAGE ONLY

    Input:
    {
        "prompt": "A cinematic vertical shot of...",
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

    input_data = job.get("input", {})

    prompt = input_data.get("prompt")
    if not prompt:
        return {"error": "Missing required parameter: prompt"}

    width = input_data.get("width", 832)
    height = input_data.get("height", 1536)
    guidance_scale = input_data.get("guidance_scale", 3.5)
    num_inference_steps = input_data.get("num_inference_steps", 28)
    seed = input_data.get("seed", None)

    # Validate dimensions (must be divisible by 16 for FLUX)
    if width % 16 != 0 or height % 16 != 0:
        return {"error": f"Dimensions must be divisible by 16. Got: {width}x{height}"}

    try:
        result = generate_image(
            prompt=prompt,
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
            "image": result,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# Start the Runpod serverless worker
runpod.serverless.start({"handler": handler})
