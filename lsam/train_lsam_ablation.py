import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry


PROJECT_DIR = "/home/shqipe/Desktop/sam_phantom_project"

IMAGE_DIR = f"{PROJECT_DIR}/SAM_10_runs/LSAM_train_dataset/images"
MASK_DIR = f"{PROJECT_DIR}/SAM_10_runs/LSAM_train_dataset/masks"

SAM_CHECKPOINT = f"{PROJECT_DIR}/checkpoints/sam_vit_b_01ec64.pth"
OUT_DIR = f"{PROJECT_DIR}/lsam_project/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 1024
MASK_SIZE = 256
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-5


class LSAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images = sorted(glob(os.path.join(image_dir, "*.png")))
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def get_box(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return np.array([0, 0, 1, 1], dtype=np.float32)

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(mask.shape[1] - 1, x2 + pad)
        y2 = min(mask.shape[0] - 1, y2 + pad)

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def get_boundary(self, mask):
        mask_uint = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint, 50, 150)
        return (edges > 0).astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        fname = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, fname)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        boundary = self.get_boundary(mask)
        box = self.get_box(mask)

        h, w = image.shape[:2]

        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask_resized = cv2.resize(mask, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_NEAREST)
        boundary_resized = cv2.resize(boundary, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_NEAREST)

        scale_x = IMG_SIZE / w
        scale_y = IMG_SIZE / h
        box[0] *= scale_x
        box[2] *= scale_x
        box[1] *= scale_y
        box[3] *= scale_y

        image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
        boundary_tensor = torch.tensor(boundary_resized).unsqueeze(0).float()
        box_tensor = torch.tensor(box).float()

        return image_tensor, mask_tensor, boundary_tensor, box_tensor


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2.0 * intersection + eps) / (union + eps)


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class LowContrastEnhancementBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.attention(x)
        return x + x * weight


class IrregularBoundaryBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.edge_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.edge_refine(x)


class MultiLesionContextBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.context(x)


class SimpleAdapter(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.adapter(x)


class LesionSpecializedAdapter(nn.Module):
    def __init__(self, variant, channels=256):
        super().__init__()

        if variant == "simple_adapter":
            self.adapter = SimpleAdapter(channels)
        else:
            self.low_contrast = IdentityBlock() if variant == "no_low_contrast" else LowContrastEnhancementBlock(channels)
            self.boundary_modeling = IdentityBlock() if variant == "no_boundary_block" else IrregularBoundaryBlock(channels)
            self.multi_lesion_context = IdentityBlock() if variant == "no_multi_context" else MultiLesionContextBlock(channels)
            self.adapter = None

    def forward(self, x):
        if self.adapter is not None:
            return self.adapter(x)

        x = self.low_contrast(x)
        x = self.boundary_modeling(x)
        x = self.multi_lesion_context(x)
        return x


class BoundaryHead(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "full",
            "no_boundary_loss",
            "no_low_contrast",
            "no_boundary_block",
            "no_multi_context",
            "simple_adapter"
        ]
    )
    args = parser.parse_args()

    variant = args.variant
    boundary_weight = 0.0 if variant == "no_boundary_loss" else 0.3

    print("Device:", DEVICE)
    print("Variant:", variant)
    print("Boundary weight:", boundary_weight)

    dataset = LSAMDataset(IMAGE_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE)

    for p in sam.image_encoder.parameters():
        p.requires_grad = False

    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    for p in sam.mask_decoder.parameters():
        p.requires_grad = True

    adapter = LesionSpecializedAdapter(variant).to(DEVICE)
    boundary_head = BoundaryHead().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(sam.mask_decoder.parameters()) +
        list(adapter.parameters()) +
        list(boundary_head.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    bce = nn.BCEWithLogitsLoss()
    best_loss = 9999

    for epoch in range(EPOCHS):
        sam.train()
        adapter.train()
        boundary_head.train()

        epoch_loss = 0.0

        for image, mask, boundary, box in tqdm(loader):
            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            boundary = boundary.to(DEVICE)
            box = box.to(DEVICE)

            image = sam.preprocess(image)

            with torch.no_grad():
                image_embedding = sam.image_encoder(image)

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box,
                    masks=None
                )

            adapted_embedding = adapter(image_embedding)

            low_res_masks, _ = sam.mask_decoder(
                image_embeddings=adapted_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            boundary_pred = boundary_head(adapted_embedding)
            boundary_pred = F.interpolate(
                boundary_pred,
                size=(MASK_SIZE, MASK_SIZE),
                mode="bilinear",
                align_corners=False
            )

            seg_loss = bce(low_res_masks, mask) + dice_loss(low_res_masks, mask)
            boundary_loss = bce(boundary_pred, boundary)

            loss = seg_loss + boundary_weight * boundary_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

            save_path = os.path.join(OUT_DIR, f"lsam_ablation_{variant}.pth")
            torch.save(
                {
                    "variant": variant,
                    "sam_mask_decoder": sam.mask_decoder.state_dict(),
                    "adapter": adapter.state_dict(),
                    "boundary_head": boundary_head.state_dict(),
                    "epoch": epoch + 1,
                    "loss": best_loss
                },
                save_path
            )

            print("Saved:", save_path)

    print("Finished:", variant)
    print("Best loss:", best_loss)


if __name__ == "__main__":
    main()
