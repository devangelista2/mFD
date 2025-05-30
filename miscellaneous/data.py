import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MayoDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        super().__init__()

        self.data_path = data_path
        self.transforms = transforms

        # We expect data_path to be like "./data/Mayo/train" or "./data/Mayo/test"
        self.fname_list = glob.glob(f"{data_path}/*/*.png")

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        # Load the idx's image from fname_list
        img_path = self.fname_list[idx]

        # To load the image as grey-scale
        x = Image.open(img_path).convert("L")
        x = transforms.ToTensor()(x)

        if self.transforms:
            return self.transforms(x)
        return x
