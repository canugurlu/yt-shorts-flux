"""
Local test script for FLUX.1-dev handler
Run this locally before deploying to Runpod
"""

import sys
import base64
from handler import handler


def save_base64_image(base64_string, filename):
    """Save base64 encoded image to file"""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    image_data = base64.b64decode(base64_string)
    with open(filename, "wb") as f:
        f.write(image_data)
    print(f"Saved: {filename}")


def main():
    # Test event
    test_event = {
        "input": {
            "prompt": "A cinematic vertical shot of a samurai standing in a neon-lit cyberpunk street, rain falling, reflections on wet pavement, highly detailed, dramatic lighting, 8k quality",
            "num_images": 2,
            "width": 832,
            "height": 1536,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "seed": 42
        }
    }

    print("=" * 60)
    print("Testing FLUX.1-dev Handler")
    print("=" * 60)
    print(f"Prompt: {test_event['input']['prompt'][:80]}...")
    print(f"Resolution: {test_event['input']['width']}x{test_event['input']['height']}")
    print(f"Number of images: {test_event['input']['num_images']}")
    print("=" * 60)

    # Run handler
    print("\nGenerating images... (this may take a while)\n")
    result = handler(test_event)

    # Print result
    if result.get("status") == "success":
        print(f"\n✓ Success! Generated {result['count']} images\n")

        # Save images
        for img_data in result.get("images", []):
            filename = f"test_output_{img_data['index']}.png"
            save_base64_image(img_data["image_base64"], filename)

        print("=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
    else:
        print(f"\n✗ Error: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
