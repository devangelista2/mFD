import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

from miscellaneous import data, augmenter
from embedder import moco_v2

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}.")

img_size = 256
batch_size = 16

# MoCov2-related Parameters
momentum_m = 0.999
temperature = 0.07
queue_size = 4096
embedding_dim = 128

# Instantiate MoCo v2 model
moco = moco_v2.MoCo(
    in_channels=1,
    embedding_dim=embedding_dim,
    K=queue_size,
    m=momentum_m,
    T=temperature,
    device=device,
).to(device)


# Load model weights
moco.load_state_dict(
    torch.load(
        os.path.join("model_weights", "mocov2.pth"),
        map_location=device,
        weights_only=True,
    ),
)
print(f"Model weights loaded.")

# t-SNE visualization of embeddings
# Prepare a simple loader without dual crops
test_data = data.MayoDataset(
    data_path=os.path.join("..", "data", "Mayo", "train"),
    transforms=augmenter.SimpleTransform(img_size),
)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Compute evaluation
moco.eval()
features = []
with torch.no_grad():
    pbar = tqdm(test_loader, desc="Computing t-SNE features")
    for imgs in pbar:
        # Send to device
        imgs = imgs.to(device)

        # Compute features via MoCo v2z
        feat = moco.encoder_q(imgs)
        feat = nn.functional.normalize(feat, dim=1)

        # Save elements on the list
        features.append(feat.cpu().numpy())
features = np.vstack(features)

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
feat_2d = tsne.fit_transform(features)

plt.figure(figsize=(8, 8))
sc = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], s=5)
plt.grid(alpha=0.3)
plt.title("t-SNE of MoCo v2 Embeddings")
plt.show()
