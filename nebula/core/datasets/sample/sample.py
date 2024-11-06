import glob
import json
import os
import numpy as np
import logging
import torch
from nebula.core.datasets.nebuladataset import NebulaDataset
from torchvision import transforms
from torch.utils.data import Dataset


CLASSES = ["2s1", "bmp2", "btr70", "m1", "m2", "m35", "m548", "m60", "t72", "zsu23"]


class Sample(Dataset):
    """
    SAMPLE dataset class.
    Dataset: https://doi.org/10.1117/12.2523460
    It is necessary to download the dataset and place it in the "data" folder in the same directory as this file.
    """

    def __init__(self, is_train=False, transform=None):
        self.is_train = is_train

        self.data = []
        self.targets = []
        self.serial_numbers = []
        self.image_list = []
        self.label_list = []

        # Path to data is "data" folder in the same directory as this file
        self.path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

        self.transform = transform

        for cls in CLASSES:
            all_measured = glob.glob("{}/{}/{}/*.png".format(self.path_to_data, "real", cls))
            for measured in all_measured:
                if not self.is_train and "elevDeg_017" in measured:
                    self.data.append([measured, CLASSES.index(cls)])
                elif self.is_train and "elevDeg_017" not in measured:
                    self.data.append([measured, CLASSES.index(cls)])
            
            self.targets += [CLASSES.index(cls)] * len(all_measured)
            self.serial_numbers += [measured.split("/")[-1].split(".")[0] for measured in all_measured]
            self.image_list += [measured for measured in all_measured]
            self.label_list += [measured.replace(".png", ".json") for measured in all_measured]

        self.data = sorted(self.data, key=lambda x: x[0])
        self.targets = sorted(self.targets)
        assert len(self.data) == len(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = np.load(self.image_list[idx])
        with open(self.label_list[idx], "r", encoding="utf-8") as f:
            label_info = json.load(f)
        _label = label_info["class_id"]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label

    def _load_metadata(self):
        self.targets = []
        self.serial_numbers = []
        for label_path in self.label_list:
            with open(label_path, "r", encoding="utf-8") as f:
                label_info = json.load(f)
            self.targets.append(label_info["class_id"])
            self.serial_numbers.append(label_info["serial_number"])

    def get_targets(self):
        if not self.targets:
            logging.info(f"Loading Metadata for {self.__class__.__name__}")
            self._load_metadata()
        return self.targets


class SampleDataset(NebulaDataset):
    def __init__(
        self,
        num_classes=10,
        partition_id=0,
        partitions_number=1,
        batch_size=32,
        num_workers=4,
        iid=True,
        partition="dirichlet",
        partition_parameter=0.5,
        seed=42,
        config=None,
    ):
        super().__init__(
            num_classes=num_classes,
            partition_id=partition_id,
            partitions_number=partitions_number,
            batch_size=batch_size,
            num_workers=num_workers,
            iid=iid,
            partition=partition,
            partition_parameter=partition_parameter,
            seed=seed,
            config=config,
        )

    def initialize_dataset(self):
        if self.train_set is None:
            self.train_set = self.load_sample_dataset(train=True)
        if self.test_set is None:
            self.test_set = self.load_sample_dataset(train=False)

        train_targets = self.train_set.get_targets()
        test_targets = self.test_set.get_targets()

        self.test_indices_map = list(range(len(self.test_set)))

        # Depending on the iid flag, generate a non-iid or iid map of the train set
        if self.iid:
            logging.info("Generating IID partition - Train")
            self.train_indices_map = self.generate_iid_map(self.train_set, self.partition, self.partition_parameter)
            logging.info("Generating IID partition - Test")
            self.local_test_indices_map = self.generate_iid_map(self.test_set, self.partition, self.partition_parameter)
        else:
            logging.info("Generating Non-IID partition - Train")
            self.train_indices_map = self.generate_non_iid_map(self.train_set, self.partition, self.partition_parameter)
            logging.info("Generating Non-IID partition - Test")
            self.local_test_indices_map = self.generate_non_iid_map(self.test_set, self.partition, self.partition_parameter)

        print(f"Length of train indices map: {len(self.train_indices_map)}")
        print(f"Lenght of test indices map (global): {len(self.test_indices_map)}")
        print(f"Length of test indices map (local): {len(self.local_test_indices_map)}")

    def load_sample_dataset(self, train=True):
        apply_transforms = [
            transforms.CenterCrop(88),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        if train:
            apply_transforms = [
                transforms.RandomCrop(88),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]

        return Sample(name="soc", is_train=train, transform=transforms.Compose(apply_transforms))

    def generate_non_iid_map(self, dataset, partition="dirichlet", partition_parameter=0.5):
        if partition == "dirichlet":
            partitions_map = self.dirichlet_partition(dataset, alpha=partition_parameter)
        elif partition == "percent":
            partitions_map = self.percentage_partition(dataset, percentage=partition_parameter)
        else:
            raise ValueError(f"Partition {partition} is not supported for Non-IID map")

        if self.partition_id == 0:
            self.plot_data_distribution(dataset, partitions_map)
            self.plot_all_data_distribution(dataset, partitions_map)

        return partitions_map[self.partition_id]

    def generate_iid_map(self, dataset, partition="balancediid", partition_parameter=2):
        if partition == "balancediid":
            partitions_map = self.balanced_iid_partition(dataset)
        elif partition == "unbalancediid":
            partitions_map = self.unbalanced_iid_partition(dataset, imbalance_factor=partition_parameter)
        else:
            raise ValueError(f"Partition {partition} is not supported for IID map")

        if self.partition_id == 0:
            self.plot_data_distribution(dataset, partitions_map)
            self.plot_all_data_distribution(dataset, partitions_map)
            
        return partitions_map[self.partition_id]