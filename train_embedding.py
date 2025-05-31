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

# Training parameters
num_epochs = 200
batch_size = 32
learning_rate = 1e-4
weights_folder = "model_weights/"
os.makedirs(weights_folder, exist_ok=True)

# MoCov2-related Parameters
momentum_m = 0.999
temperature = 0.07
queue_size = 4096
embedding_dim = 128

# --- Load data (with augmentation) ---
train_dataset = data.MayoDataset(
    data_path=os.path.join("..", "data", "Mayo", "train"),
    transforms=augmenter.TwoCropsTransform(
        img_size=img_size,
        rotation_angle=15,
        crop_scale=0.9,
        noise_level=0.01,
    ),
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

# Instantiate MoCo v2 model
moco = moco_v2.MoCo(
    in_channels=1,
    embedding_dim=embedding_dim,
    K=queue_size,
    m=momentum_m,
    T=temperature,
    device=device,
).to(device)

# Setup optimizer and loss function
optimizer = optim.Adam(moco.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training loop
moco.train()
for epoch in range(num_epochs):

    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs}", leave=True)
    for t, (im_q, im_k) in enumerate(pbar):
        # Send data to device
        im_q, im_k = im_q.to(device), im_k.to(device)

        # Che moco output
        logits, labels = moco(im_q, im_k)

        # Compute InfoNCE Loss
        optimizer.zero_grad()
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Update loss
        total_loss += loss.item()

        # Update tqdm bar
        pbar.set_postfix(
            {
                "Loss": f"{total_loss / (t + 1):.4f}",
            }
        )

    # Save model every 5 epochs (overwrite)
    if (epoch + 1) % 5 == 0:
        torch.save(moco.state_dict(), os.path.join(weights_folder, "mocov2.pth"))
        print(f"Checkpoint saved at epoch {epoch+1}.")

# t-SNE visualization of embeddings
# Prepare a simple loader without dual crops
test_data = data.MayoDataset(
    data_path=os.path.join("..", "data", "Mayo", "test"),
    transforms=augmenter.SimpleTransform(img_size),
)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Compute evaluation
moco.eval()
features, test_labels = [], []
with torch.no_grad():
    for imgs, labs in test_loader:
        # Send to device
        imgs = imgs.to(device)

        # Compute features via MoCo v2z
        feat = moco.encoder_q(imgs)
        feat = nn.functional.normalize(feat, dim=1)

        # Save elements on the list
        features.append(feat.cpu().numpy())
        test_labels.extend(labs.numpy())
features = np.vstack(features)

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
feat_2d = tsne.fit_transform(features)

plt.figure(figsize=(8, 8))
sc = plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=test_labels, s=5, cmap="tab10")
plt.legend(*sc.legend_elements(), title="Classes")
plt.title("t-SNE of MoCo v2 Embeddings")
plt.show()
