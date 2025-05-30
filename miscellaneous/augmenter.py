from torchvision import transforms
import torch


# Two crops transformation for contrastive learning
class TwoCropsTransform:
    def __init__(
        self,
        img_size,
        rotation_angle=15,
        crop_scale=0.8,
        noise_level=0.01,
    ):
        self.base_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomResizedCrop(img_size, scale=(crop_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(rotation_angle),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x + noise_level * torch.randn_like(x)),
            ]
        )

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


class SimpleTransform:
    def __init__(
        self,
        img_size,
    ):
        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __call__(self, x):
        return self.transform(x)
