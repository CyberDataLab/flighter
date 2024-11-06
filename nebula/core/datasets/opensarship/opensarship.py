import glob
import json
import os
import shutil
import zipfile
import numpy as np
import logging
import torch
from nebula.core.datasets.nebuladataset import NebulaDataset
from torchvision import transforms
from torch.utils.data import Dataset


class OpenSarShip(Dataset):
    """
    OpenSarShip dataset class.
    Dataset: https://doi.org/10.1109/JSTARS.2017.2755672
    It extracts the data, processes it, and loads it into PyTorch format.
    """

    def __init__(self, archive_path, dest_folder="opensar_data", chosen_classes=["Cargo", "Tanker", "Fishing"], is_train=False, transform=None):
        """
        Initializes the OpenSarShip dataset by extracting, processing, and loading the data.

        Args:
            archive_path (str): Path to the OpenSARShip dataset archive (zip file).
            dest_folder (str): Destination folder to extract and store processed data.
            chosen_classes (list): Classes to filter and load.
            is_train (bool): Whether to load the training or testing dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.is_train = is_train
        self.name = "opensar_data"  # Hardcoded for simplicity
        self.chosen_classes = chosen_classes
        self.data = []
        self.targets = []
        self.serial_numbers = []
        self.path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), dest_folder)
        self.transform = transform

        # Extract and process the dataset
        self._extract_and_process_data(archive_path, dest_folder)

        # Load the dataset after processing
        mode = "train" if self.is_train else "test"
        self.image_list = glob.glob(os.path.join(self.path_to_data, f"{mode}/*/*.npy"))
        self.label_list = glob.glob(os.path.join(self.path_to_data, f"{mode}/*/*.json"))
        self.image_list = sorted(self.image_list, key=os.path.basename)
        self.label_list = sorted(self.label_list, key=os.path.basename)

        assert len(self.image_list) == len(self.label_list)

        self._load_metadata()

    def _extract_and_process_data(self, archive_path, dest_folder):
        """
        Extracts the dataset from the archive and processes it, filtering out chosen classes.

        Args:
            archive_path (str): Path to the OpenSARShip archive (zip file).
            dest_folder (str): Destination folder for extracted and processed data.
        """
        # Create destination folder if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)

        # Extract the archive
        with zipfile.ZipFile(archive_path, 'r') as main_archive:
            for member in main_archive.namelist():
                if member.endswith('.zip'):
                    with zipfile.ZipFile(main_archive.open(member), 'r') as nested_archive:
                        self._process_nested_archive(nested_archive, dest_folder)
                elif member.startswith('Patch_RGB/'):
                    main_archive.extract(member, dest_folder)

        # Process the extracted dataset to keep only the chosen classes
        self._process_classes(dest_folder)

    def _process_nested_archive(self, nested_archive, destination_folder):
        """
        Extracts relevant files from a nested zip archive.

        Args:
            nested_archive: The opened nested zip archive.
            destination_folder: Destination folder for the extracted data.
        """
        for nested_member in nested_archive.namelist():
            if nested_member.startswith('Patch_RGB/'):
                nested_archive.extract(nested_member, destination_folder)

    def _process_classes(self, src_folder):
        """
        Filters and processes images from chosen classes and copies them to appropriate folders.

        Args:
            src_folder (str): Source folder containing extracted data.
        """
        src_folder += "/Patch_RGB"
        stats = self._load_data_and_stats(src_folder)

        for key, value in stats.items():
            if key in self.chosen_classes:
                class_folder = os.path.join(self.path_to_data, key)
                os.makedirs(class_folder, exist_ok=True)

                for filename in value['unique_filenames']:
                    if filename.endswith("vv.png"):  # Process only vv.png files
                        source_path = os.path.join(src_folder, filename)
                        destination_path = os.path.join(class_folder, filename)
                        shutil.copy2(source_path, destination_path)

    def _load_data_and_stats(self, folder_path):
        """
        Loads statistics and file data from the folder for chosen classes.

        Args:
            folder_path (str): Path to the folder containing data files.

        Returns:
            dict: A dictionary with statistics and filenames for each class.
        """
        class_stats = {}
        for filename in os.listdir(folder_path):
            class_name = filename.split("_")[0]  # Assuming first word before _ is the class
            if class_name not in class_stats:
                class_stats[class_name] = {"count": 0, "unique_filenames": []}
            class_stats[class_name]["count"] += 1
            class_stats[class_name]["unique_filenames"].append(filename)
        return class_stats

    def _load_metadata(self):
        """
        Loads the metadata (targets and serial numbers) from the label files.
        """
        self.targets = []
        self.serial_numbers = []
        for label_path in self.label_list:
            with open(label_path, "r", encoding="utf-8") as f:
                label_info = json.load(f)
            self.targets.append(label_info["class_id"])
            self.serial_numbers.append(label_info["serial_number"])

    def get_targets(self):
        """
        Returns the targets (class IDs) for the dataset.

        Returns:
            list: A list of class IDs for the dataset.
        """
        if not self.targets:
            logging.info(f"Loading Metadata for {self.__class__.__name__}")
            self._load_metadata()
        return self.targets

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.targets)

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding label for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label) for the given index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _image = np.load(self.image_list[idx])
        with open(self.label_list[idx], "r", encoding="utf-8") as f:
            label_info = json.load(f)
        _label = label_info["class_id"]

        if self.transform:
            _image = self.transform(_image)

        return _image, _label
    

class OpenSarShipDataset(NebulaDataset):
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
            self.train_set = self.load_opensarship_dataset(train=True)
        if self.test_set is None:
            self.test_set = self.load_opensarship_dataset(train=False)

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

    def load_opensarship_dataset(self, train=True):
        apply_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        if train:
            apply_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

        return OpenSarShip(name="soc", is_train=train, transform=transforms.Compose(apply_transforms))

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
