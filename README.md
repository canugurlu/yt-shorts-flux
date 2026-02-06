# YouTube Shorts FLUX T2I - Runpod Serverless

FLUX.1-dev modeli ile YouTube Shorts için yüksek kaliteli görsel üreten Runpod Serverless template.

## Özellikler

- **Model**: FLUX.1-dev (Black Forest Labs)
- **Resolution**: 832×1536 (9:16 aspect ratio - YouTube Shorts için ideal)
- **Format**: PNG (lossless, I2V için uygun)
- **GPU**: A40 (24GB VRAM) veya üstü
- **Output**: Base64 encoded görseller

## Hızlı Başlangıç

### 1. Repository'yi Klonla

```bash
git clone https://github.com/canugurlu/yt-shorts-flux.git
cd yt-shorts-flux
```

### 2. Runpod Container Registry'e Push Et

```bash
# Login
runpodctl login

# Build ve push
runpodctl build docker -f Dockerfile -t yt-shorts-flux:latest
```

### 3. Serverless Template Oluştur

Runpod Console'da:
1. **Serverless** → **Templates** → **New Template**
2. Container Registry'den image'ı seç
3. Name: `yt-shorts-flux`
4. Min/Max CPUs: 1-2
5. Min/Max GPUs: 1
6. GPU: A40 (24GB) veya RTX 6000 Ada
7. Timeout: 120s
8. Environment Variables:
   - `MODEL_ID`: `black-forest-labs/FLUX.1-dev`
   - `HANDLER`: `handler.py`

### 4. Endpoint Oluştur

1. **Serverless** → **Deployments** → **New Deployment**
2. Template seç: `yt-shorts-flux`
3. Idle timeout: 300s (beş dakika)
4. Workers: 1-5

## API Kullanımı

### Request

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "A cinematic vertical shot of a futuristic city at sunset, cyberpunk style, neon lights, highly detailed",
    "num_images": 5,
    "width": 832,
    "height": 1536,
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "seed": 42
  }'
```

### Response

```json
{
  "status": "success",
  "prompt": "...",
  "model": "black-forest-labs/FLUX.1-dev",
  "count": 5,
  "images": [
    {
      "index": 0,
      "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
      "width": 832,
      "height": 1536
    }
  ]
}
```

## Parametreler

| Parametre | Tip | Default | Açıklama |
|-----------|-----|---------|----------|
| `prompt` | string | - | Gerekli. Görsel tarifi |
| `num_images` | int | 5 | Kaç görsel üretilecek |
| `width` | int | 832 | Genişlik (16'nın katı olmalı) |
| `height` | int | 1536 | Yükseklik (16'nın katı olmalı) |
| `guidance_scale` | float | 3.5 | Prompt adherence (1.0-10.0) |
| `num_inference_steps` | int | 28 | Adet sayısı (kalite için 28-50) |
| `seed` | int | null | Tekrarlanabilirlik için |

## Maliyet Tahmini

| GPU | Cost/saat | 5 Görsel Tahmini |
|-----|-----------|------------------|
| RTX 4000 Ada | $0.32 | ~$0.03-0.05 |
| A40 | $0.79 | ~$0.08-0.12 |

## GitHub Actions ile Otomatik Deploy

`.github/workflows/deploy.yml` dosyası her push'ta otomatik build eder.

```yaml
name: Deploy to Runpod

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push to Runpod
        run: |
          runpodctl build docker -f Dockerfile -t yt-shorts-flux:latest
```

## Notlar

- İlk request (cold start) ~15-20 saniye sürebilir (model yükleme)
- Sonraki request'ler ~8-10 saniye
- Model weights cache'lenir, tekrar tekrar indirilmez
- Base64 output direkt Wan2.2 I2V'e verilebilir

## Lisans

FLUX.1-dev lisans şartlarına uygun kullanın.
