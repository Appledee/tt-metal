import sys
import os
from PIL import Image
import torchvision.transforms as T

# Add the parent 'tt' folder to the path so we can import the model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tt'))

from tt_clip_ViT_B_32 import forward  # import your TTNN model

def run_vit_on_image(image_path="../image/sample.jpg"):
    import ttnn

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at path: {image_path}")

    print("📷 Loading image...")
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

    print("📤 Converting to TTNN tensor...")
    ttnn_input = ttnn.from_torch(image_tensor)

    print("⚙️ Running forward pass through ViT model...")
    output = forward(ttnn_input)

    print("📥 Converting output to PyTorch tensor...")
    embedding = ttnn.to_torch(output)

    print("✅ CLIP Embedding shape:", embedding.shape)
    print("🔢 First 10 values:", embedding[0, :50])

    return embedding

if __name__ == "__main__":
    try:
        run_vit_on_image()
    except Exception as e:
        print("❌ Error during inference:", e)
