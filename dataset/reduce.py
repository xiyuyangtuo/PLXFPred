import pickle
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict
import os
from dataset import ProteinLigandData

# Log configuration
log_dir = "/logs/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "dataset_diff.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(log_file_path, maxBytes=10**6, backupCount=5)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# File paths
full_dataset_path = "pdb2020_dataset_diff.pkl"
subset_dataset_path = "CASF_core2013.pkl"
output_dataset_path = "pdb2020_dataset.pkl"

# Load dataset
def load_pickle(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load file: {file_path}, Error: {e}")
        return []

# Save dataset
def save_pickle(data: List[Dict], file_path: str) -> None:
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Dataset saved to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {file_path}, Error: {e}")

# Main function
if __name__ == "__main__":
    logger.info("Starting to process the dataset...")

    # Load datasets
    full_dataset = load_pickle(full_dataset_path)
    subset_dataset = load_pickle(subset_dataset_path)

    if not full_dataset or not subset_dataset:
        logger.error("Loaded datasets are empty, cannot proceed.")
        exit(1)

    # Extract PDB names from the subset dataset
    subset_pdb_names = {item['id'] for item in subset_dataset if 'id' in item}

    # Remove entries from the full dataset that are in the subset dataset
    diff_dataset = [item for item in full_dataset if item['id'] not in subset_pdb_names]

    # Save the result
    save_pickle(diff_dataset, output_dataset_path)

    logger.info(f"Processing completed, the final dataset contains {len(diff_dataset)} entries.")