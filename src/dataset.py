
import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH
from src.preprocess import preprocess_thermal


class SolarThermalDataset(Dataset):

    def __init__(self, csv_path, use_clahe=False, train=False):
        self.data = pd.read_csv(csv_path).reset_index(drop=True)

        self.use_clahe = use_clahe
        self.train = train

        required_columns = {"path", "y"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                f"CSV must contain columns: {required_columns}"
            )

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(4, 4)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["path"]
        label = int(row["y"])

        # Read grayscale thermal image
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        # Apply preprocessing
        image_tensor = preprocess_thermal(
            image,
            use_clahe=self.use_clahe
        )

        # Convert NumPy array to PyTorch tensor
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_tensor)
        ).float()

        # Light augmentation: training only
        if self.train:

            if np.random.rand() < 0.5:
                image_tensor = torch.flip(
                    image_tensor, dims=[2]
                )

            if np.random.rand() < 0.5:
                image_tensor = torch.flip(
                    image_tensor, dims=[1]
                )

        return image_tensor, label
