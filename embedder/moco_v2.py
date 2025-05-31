import torch
from torch import nn

from torchvision import models


# MoCo v2 model definition
class MoCo(nn.Module):
    def __init__(
        self,
        in_channels,
        embedding_dim=128,
        K=4096,
        m=0.999,
        T=0.07,
        device="cpu",
    ):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.device = device

        # create the encoders
        self.encoder_q = get_encoder(in_channels, embedding_dim, device)
        self.encoder_k = get_encoder(in_channels, embedding_dim, device)
        self._init_encoders()

        # create the queue
        self.register_buffer("queue", torch.randn(embedding_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _init_encoders(self):
        # initialize key encoder to query encoder weights
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        # compute query features
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():
            # momentum update for key encoder
            for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.mm(q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, "Queue size must be divisible by batch size"
        # replace the keys at ptr
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


# Helper to build a ResNet-50 based encoder for 1-channel inputs
def get_encoder(
    in_channels,
    embedding_dim,
    device,
):
    model = models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    dim_mlp = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp),
        nn.ReLU(inplace=True),
        nn.Linear(dim_mlp, embedding_dim),
    )
    return model.to(device)
