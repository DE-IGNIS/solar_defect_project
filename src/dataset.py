import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocess import preprocess_thermal


class SolarThermalDataset(Dataset):

    def __init__(self, csv_path, use_clahe=False, train=False):
        self.data = pd.read_csv(csv_path, skipinitialspace=True)

        # Normalize CSV headers and text fields
        self.data.columns = self.data.columns.str.strip()
        self.data["path"] = self.data["path"].str.strip()
        self.data["anomaly_class"] = self.data["anomaly_class"].str.strip()

        self.use_clahe = use_clahe
        self.train = train

        required = {"path", "y"}
        if not required.issubset(self.data.columns):
            raise ValueError(f"Missing columns: {required - set(self.data.columns)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Unable to read: {row['path']}")

        image = preprocess_thermal(
            image,
            use_clahe=self.use_clahe
        )

        image = torch.from_numpy(
            np.ascontiguousarray(image)
        ).float()

        if self.train:
            if np.random.rand() < 0.5:
                image = torch.flip(image, dims=[2])

            if np.random.rand() < 0.5:
                image = torch.flip(image, dims=[1])

        return image, int(row["y"])