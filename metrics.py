import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from scipy import linalg


class medicalFrechetDistance(nn.Module):
    def __init__(self, embedding_model, batch_size=16, eps=1e-6, device="cpu"):
        self.eps = eps
        self.embedding_model = embedding_model

        self.batch_size = batch_size
        self.device = device

    def _get_embeddings(self, data, batch_size=32, device="cpu"):
        """
        Extract normalized embeddings from the MoCo v2 encoder for all images in data_dir.
        Assumes model.encoder_q returns the raw embedding.
        """
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        self.embedding_model.eval()
        embeddings = []
        with torch.no_grad():
            for imgs in loader:
                # Send imgs to device
                imgs = imgs.to(device)

                # Compute embedding using trained model
                feat = self.embedding_model.encoder_q(imgs)
                feat = torch.nn.functional.normalize(feat, dim=1)
                embeddings.append(feat.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def _compute_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        Compute the Frechet distance between two Gaussians with means mu1, mu2 and covariances sigma1, sigma2.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * self.eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fd = (
            diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        )
        return float(fd)

    def __call__(self, gen_data, real_data):
        # Extract embeddings
        gen_emb = self._get_embeddings(gen_data, self.batch_size, self.device)
        real_emb = self._get_embeddings(real_data, self.batch_size, self.device)

        # Compute mean and std of embeddings
        mu_r, sigma_r = real_emb.mean(axis=0), np.cov(real_emb, rowvar=False)
        mu_g, sigma_g = gen_emb.mean(axis=0), np.cov(gen_emb, rowvar=False)
        return self._compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


class medicalKernelDistance(nn.Module):
    def __init__(
        self,
        embedding_model,
        batch_size=16,
        degree=3,
        gamma=None,
        coef0=1,
        unbiased=True,
        device="cpu",
    ):
        self.embedding_model = embedding_model

        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.unbiased = unbiased

        self.batch_size = batch_size
        self.device = device

    def _compute_kernel_distance(self, x, y):
        """
        Compute MMD between samples x, y using a polynomial kernel k(a,b)=(gamma * a.b + coef0)^degree.
        Returns the (unbiased) estimator.
        """
        if self.gamma is None:
            self.gamma = 1.0 / x.shape[1]

        K_xx = (self.gamma * x.dot(x.T) + self.coef0) ** self.degree
        K_yy = (self.gamma * y.dot(y.T) + self.coef0) ** self.degree
        K_xy = (self.gamma * x.dot(y.T) + self.coef0) ** self.degree

        m = x.shape[0]
        n = y.shape[0]

        if self.unbiased:
            # remove diagonal
            sum_xx = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1))
            sum_yy = (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1))
        else:
            sum_xx = K_xx.sum() / (m * m)
            sum_yy = K_yy.sum() / (n * n)
        sum_xy = K_xy.sum() / (m * n)

        mmd = sum_xx + sum_yy - 2 * sum_xy
        return float(mmd)

    def _get_embeddings(self, data, batch_size=32, device="cpu"):
        """
        Extract normalized embeddings from the MoCo v2 encoder for all images in data_dir.
        Assumes model.encoder_q returns the raw embedding.
        """
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        self.embedding_model.eval()
        embeddings = []
        with torch.no_grad():
            for imgs in loader:
                # Send imgs to device
                imgs = imgs.to(device)

                # Compute embedding using trained model
                feat = self.embedding_model.encoder_q(imgs)
                feat = torch.nn.functional.normalize(feat, dim=1)
                embeddings.append(feat.cpu().numpy())
        embeddings = np.vstack(embeddings)
        return embeddings

    def __call__(self, gen_data, real_data):
        # Extract embeddings
        gen_emb = self._get_embeddings(gen_data, self.batch_size, self.device)
        real_emb = self._get_embeddings(real_data, self.batch_size, self.device)

        # Compute Kernel Distance
        return self._compute_kernel_distance(
            gen_emb,
            real_emb,
            degree=3,
            gamma=None,
            coef0=1,
            unbiased=True,
        )
