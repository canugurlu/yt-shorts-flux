# CLAUDE.md - Runpod Serverless FLUX.1-dev Project Notes

## Project Overview
- **Model**: black-forest-labs/FLUX.1-dev (gated Hugging Face model)
- **Purpose**: YouTube Shorts text-to-image generation (9:16 vertical format)
- **Platform**: Runpod Serverless
- **Docker Image**: `jaxnlindemann/yt-shorts-flux:latest`
- **GPU**: A40 (24GB VRAM)

## Critical Runpod Serverless Requirements

### Handler Structure (MANDATORY)
Every Runpod serverless handler **MUST** have these components:

```python
import runpod  # REQUIRED

def handler(job):  # Use 'job' parameter, not 'event'
    """
    job["input"] contains the input parameters
    """
    input_data = job.get("input", {})
    # ... processing ...
    return result

# CRITICAL - This MUST be at the end of handler.py
runpod.serverless.start({"handler": handler})
```

### Common Mistakes to Avoid

1. **Missing `runpod.serverless.start()`** - Worker will NOT start
2. **Wrong parameter name** - Use `job` not `event` (though both work, `job` is standard)
3. **Accessing input wrong** - Use `job["input"]` or `job.get("input", {})`
4. **Not installing runpod package** - Add `pip install runpod` to Dockerfile

## Known Issues & Solutions

### 1. Gated Model Access
**Error**: `Gated model access denied`
**Solution**: Pass `HF_TOKEN` as build arg in Dockerfile and GitHub Actions

### 2. Disk Space
**Error**: `No space left on device`
**Solution**: Set container disk to 50GB+ in Runpod template settings

### 3. CUDA OOM / VRAM Issues
**Current approach**: Quality mode - GPU-only optimizations, no CPU offload
- Resolution: 832x1536 (9:16 vertical) - script'ten override edilebilir
- bfloat16 dtype
- VAE slicing + tiling
- Attention slicing

#### VRAM Optimization Techniques (handler.py)

**Currently Enabled (Quality-Preserving):**
```python
# VAE optimizations - ~2GB savings, no quality impact
pipeline.vae.enable_slicing()
pipeline.vae.enable_tiling()

# Attention slicing - ~1-2GB savings, no quality impact
pipeline.enable_attention_slicing()
```

**Disabled (not working with FLUX):**
- `BitsAndBytesConfig` - Causes `PipelineQuantizationConfig` error, REMOVED
- `transformer.enable_gradient_checkpointing()` - Not supported by FLUX transformer
- CPU offload - Not available on Runpod serverless (no CPU)

**Quality vs VRAM Trade-offs:**

| Technique | VRAM Savings | Quality Impact | Speed Impact | Status |
|-----------|--------------|----------------|--------------|--------|
| VAE Slicing | ~1GB | None | Slight slower | ✅ ENABLED |
| VAE Tiling | ~1GB | None | Slight slower | ✅ ENABLED |
| Attention Slicing | ~1-2GB | None | Slower | ✅ ENABLED |
| bfloat16 | ~30% vs fp32 | None | Faster | ✅ ENABLED |
| Lower Resolution (640x1152) | ~30% | Minimal | Faster | ⚙️ OPTIONAL |
| FP16 VAE | ~1-2GB | Minimal | Faster | ⚙️ OPTIONAL |

**Expected VRAM Usage:**
- 832x1536: ~18-20GB
- 640x1152: ~14-16GB
- Target GPU: A40 (24GB)

### 4. Docker Hub Authentication
**Error**: Failed login to Docker Hub
**Solution**: Set `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` in GitHub Secrets

## Input Format

```json
{
  "input": {
    "prompt": "A cinematic vertical shot of...",
    "width": 832,
    "height": 1536,
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "seed": 42
  }
}
```

## Output Format

```json
{
  "status": "success",
  "prompt": "...",
  "model": "black-forest-labs/FLUX.1-dev",
  "image": {
    "image_base64": "base64 encoded PNG",
    "width": 832,
    "height": 1536
  }
}
```

## Resolution Constraints
- FLUX requires dimensions divisible by 16
- Recommended for YouTube Shorts: 832x1536 (9:16)
- Other valid sizes: 768x1344, 896x1600, etc.

## References
- Runpod Serverless Docs: https://docs.runpod.io/serverless/workers/handler-functions
- FLUX.1 Model: https://huggingface.co/black-forest-labs/FLUX.1-dev
