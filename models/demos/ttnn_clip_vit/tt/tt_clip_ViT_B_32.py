# ✅ Step 1: Load Pretrained PyTorch Weights
import torch
import ttnn

import os
import torch

model_path = os.path.join(os.path.dirname(__file__), "ViT-B-32.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"\n[ERROR] Could not find model weights at: {model_path}\n"
        "Please download 'ViT-B-32.pt' and place it in the same folder as tt_clip_ViT_B_32.py.\n"
        "You can generate it with:\n"
        "  import torch\n"
        "  from transformers import CLIPModel\n"
        "  model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')\n"
        "  torch.save(model.vision_model.state_dict(), 'ViT-B-32.pt')"
    )

state_dict = torch.load(model_path, map_location="cpu")


# ✅ Step 2: Extract Model Dimensions
vision_width = state_dict["visual.conv1.weight"].shape[0]  # usually 768
patch_size = state_dict["visual.conv1.weight"].shape[-1]   # 32
num_blocks = len([k for k in state_dict if k.endswith(".attn.in_proj_weight")])

# ✅ Step 3: Patch Embedding Layer in TTNN
conv1 = ttnn.Conv2D(
    in_channels=3,
    out_channels=vision_width,
    kernel_size=patch_size,
    stride=patch_size,
    bias=False
)
conv1.weight = ttnn.from_torch(state_dict["visual.conv1.weight"])

# ✅ Step 4: Add Class Token + Positional Embedding
cls_token = ttnn.Parameter(ttnn.from_torch(state_dict["visual.class_embedding"]).unsqueeze(0).unsqueeze(0))
pos_embed = ttnn.Parameter(ttnn.from_torch(state_dict["visual.positional_embedding"]).unsqueeze(0))

# ✅ Step 5: Pre-Norm (Before Transformer)
def apply_ln_pre(x):
    return ttnn.layer_norm(
        input_tensor=x,
        epsilon=1e-5,
        weight=ttnn.from_torch(state_dict["visual.ln_pre.weight"]),
        bias=ttnn.from_torch(state_dict["visual.ln_pre.bias"])
    )

# ✅ Step 6: Transformer Block Template
def build_transformer_block(i):
    block = ttnn.TransformerBlock(
        hidden_dim=vision_width,
        num_heads=vision_width // 64,
        mlp_hidden_dim=vision_width * 4,
        name=f"visual.transformer.resblocks.{i}"
    )
    # You would load weights for attention and MLP layers here
    return block

# ✅ Step 7: Post-Norm (After Transformer)
def apply_ln_post(x):
    return ttnn.layer_norm(
        input_tensor=x,
        epsilon=1e-5,
        weight=ttnn.from_torch(state_dict["visual.ln_post.weight"]),
        bias=ttnn.from_torch(state_dict["visual.ln_post.bias"])
    )

# ✅ Step 8: Final Projection
def project_to_clip_embedding(x):
    proj_weight = ttnn.from_torch(state_dict["visual.proj"].T)
    return ttnn.matmul(x, proj_weight)

# 🧹 Final: Assemble the Model

def forward(input_image):
    x = conv1(input_image)  # patch embedding
    B, C, H, W = x.shape
    x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N, D]

    # Add class token and positional embedding
    cls = cls_token.expand(B, -1, -1)  # [B, 1, D]
    x = ttnn.concat([cls, x], dim=1)
    x = x + pos_embed

    x = apply_ln_pre(x)
    for i in range(num_blocks):
        x = build_transformer_block(i)(x)
    x = apply_ln_post(x)

    x = x[:, 0, :]  # take CLS token
    x = project_to_clip_embedding(x)
    return x
