import os
import pickle
import random
from typing import List
from torch_geometric.data import Data
import logging
from dataset import ProteinLigandData

# Log configuration
log_dir = "/logs/dataset_logs/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "split_dataset.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load dataset
def load_dataset(file_path: str) -> List[Data]:
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"Dataset file does not exist: {file_path}")
        return []

    # Check if the file is a pickle file
    if not file_path.endswith(('.pkl', '.pickle')):
        logger.error(f"File {file_path} is not a pickle file (expected .pkl or .pickle extension)")
        return []

    try:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        logger.info(f"Successfully loaded dataset: {file_path}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []

# Save dataset
def save_dataset(dataset: List[Data], file_path: str) -> None:
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset saved to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")

# Split dataset
def split_dataset(dataset: List[Data], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple:
    # Validate that the sum of ratios is 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        logger.error(f"The sum of split ratios must be 1, currently {total_ratio}")
        return [], [], []

    # Get the size of the dataset
    dataset_size = len(dataset)
    if dataset_size == 0:
        logger.error("The dataset is empty and cannot be split!")
        return [], [], []

    # Shuffle the dataset randomly
    random.seed(42)  # Set random seed for reproducibility
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    logger.info(f"The dataset has been shuffled randomly, containing {dataset_size} samples")

    # Calculate split points
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size  # Ensure the total count is consistent

    # Split the dataset
    train_set = shuffled_dataset[:train_size]
    val_set = shuffled_dataset[train_size:train_size + val_size]
    test_set = shuffled_dataset[train_size + val_size:]

    # Log the split results
    logger.info(f"Training set: {len(train_set)} samples ({train_ratio*100:.1f}%)")
    logger.info(f"Validation set: {len(val_set)} samples ({val_ratio*100:.1f}%)")
    logger.info(f"Test set: {len(test_set)} samples ({test_ratio*100:.1f}%)")

    return train_set, val_set, test_set

# Main function
if __name__ == "__main__":
    # Dataset path
    dataset_path = "pdb2020_dataset_full.pkl"
    output_dir = " "

    # Load dataset
    dataset = load_dataset(dataset_path)
    if not dataset:
        logger.error("The dataset is empty and cannot proceed!")
        exit(1)

    # Split dataset
    train_set, val_set, test_set = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Check the split results
    if not train_set or not val_set or not test_set:
        logger.error("Failed to split the dataset, please check the logs!")
        exit(1)

    # Save the split datasets
    save_dataset(train_set, os.path.join(output_dir, "train_set.pkl"))
    save_dataset(val_set, os.path.join(output_dir, "val_set.pkl"))
    save_dataset(test_set, os.path.join(output_dir, "test_set.pkl"))

    logger.info("Dataset splitting and saving completed!")