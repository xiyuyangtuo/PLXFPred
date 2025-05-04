import os
import pickle
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Tuple, List
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
import esm
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Log configuration
log_dir = "/dataset_logs/"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "data_errors.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(log_file_path, maxBytes=10**6, backupCount=5)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Path definitions
index_file_path = ".csv"
output_dir = " "

# Import processing functions
from pro_feat import process_protein_sequence  # Import from pro_feat.py
from lig_feat import JTVAEDataPreprocessor  # Import from lig_feat.py

def clean_protein_sequence(sequence):
    allowed_chars = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned_sequence = ''.join([char for char in sequence if char in allowed_chars])
    return cleaned_sequence

# Validate if SMILES is valid
def is_valid_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)  # Remove leading/trailing whitespace
        return mol is not None
    except Exception as e:
        logger.debug(f"Failed to parse SMILES: {smiles}, error: {e}")
        return False

# Custom data class
class ProteinLigandData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seq = kwargs.get("seq", None)
        # Initialize to None only if not present in kwargs to avoid overwriting existing values
        if "prot_amino_acid_sequence" not in kwargs:
            self.prot_amino_acid_sequence = None
        if "lig_smiles" not in kwargs:
            self.lig_smiles = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'mol_edge_index':
            return self.mol_x.size(0)
        elif key == 'clique_edge_index':
            return self.clique_x.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]]).to(device)
        elif key == 'prot_edge_index':
            return self.prot_node_aa.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# Save dataset to pickle
def save_to_pickle(dataset: List[Dict], output_dir: str) -> None:
    output_file_path = os.path.join(output_dir, "predict_dataset_full.pkl")
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset saved to: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")

# Custom dataset class
class ProteinLigandDataset(Dataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        super(ProteinLigandDataset, self).__init__('.', transform, pre_transform)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

# Parse index file
def parse_index_file(file_path: str) -> List[Tuple]:
    try:
        # Read CSV, assuming the first row is the header
        df = pd.read_csv(file_path)
        logger.info(f"Detected columns: {list(df.columns)}")
        
        # Check necessary fields: only Amino_Acid_Sequence and ligand_smiles are required
        required_cols = ["Amino_Acid_Sequence", "ligand_smiles"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing necessary fields: {required_cols}")
            return []
        
        # Map columns
        amino_acid_idx = df.columns.get_loc("Amino_Acid_Sequence")
        ligand_smiles_idx = df.columns.get_loc("ligand_smiles")
        id_idx = df.columns.get_loc("id")
        
        # Regression and classification labels are optional
        regression_label_idx = df.columns.get_loc("regression_label") if "regression_label" in df.columns else None
        classification_label_idx = df.columns.get_loc("classification_label") if "classification_label" in df.columns else None
        
        # Print some data for debugging
        logger.info(f"Number of columns parsed: {len(df.columns)}")
        logger.info(f"First 5 rows of data: \n{df.head().to_string()}")
        
        # If there is a regression_label column, convert to numeric type
        if regression_label_idx is not None:
            df.iloc[:, regression_label_idx] = pd.to_numeric(df.iloc[:, regression_label_idx], errors='coerce')
        
        # If there is a classification_label column, convert to numeric type and validate
        if classification_label_idx is not None:
            df.iloc[:, classification_label_idx] = pd.to_numeric(df.iloc[:, classification_label_idx], errors='coerce', downcast='integer')
            invalid_class = df.iloc[:, classification_label_idx].notna() & ~df.iloc[:, classification_label_idx].isin([0, 1])
            if invalid_class.any():
                logger.warning(f"Found {invalid_class.sum()} invalid classification_label values (must be 0 or 1), setting to NaN")
                df.loc[invalid_class, classification_label_idx] = np.nan
        
        # Check if Amino_Acid_Sequence is a valid string
        invalid_sequences = df.iloc[:, amino_acid_idx].apply(lambda x: not isinstance(x, str) or not x.strip())
        if invalid_sequences.any():
            invalid_ids = df.iloc[:, id_idx][invalid_sequences].tolist()
            logger.warning(f"Found {invalid_sequences.sum()} invalid Amino_Acid_Sequence, involving: {invalid_ids}")
            df = df[~invalid_sequences].reset_index(drop=True)
        
        # Check validity of SMILES
        invalid_smiles = df.iloc[:, ligand_smiles_idx].apply(lambda x: not is_valid_smiles(str(x) if pd.notna(x) else ''))
        if invalid_smiles.any():
            invalid_ids = df.iloc[:, id_idx].tolist()
            invalid_ids = [pid for i, pid in enumerate(invalid_ids) if invalid_smiles.iloc[i]]
            logger.warning(f"Found {invalid_smiles.sum()} invalid SMILES, involving: {invalid_ids}")
            df = df[~invalid_smiles].reset_index(drop=True)
        else:
            logger.info("All SMILES are valid")
        
        # Construct return data
        def map_row(row):
            return (
                row[id_idx],
                row[regression_label_idx] if regression_label_idx is not None else np.nan,
                row[amino_acid_idx],
                row[classification_label_idx] if classification_label_idx is not None else np.nan,
                row[ligand_smiles_idx]
            )
        
        index_data = [map_row(row) for row in df.itertuples(index=False)]
        # Debug: Print first few rows of index_data
        logger.info(f"First 3 rows of index_data: {index_data[:3]}")
        return index_data
    except Exception as e:
        logger.error(f"Failed to parse index file: {e}")
        return []

# Process a single PDB
def process_single_pdb(amino_acid_sequence, ligand_smiles, id=None, regression_label=np.nan, classification_label=np.nan, ligand_preprocessor=None, protein_data=None) -> Optional[ProteinLigandData]:
    try:
        if not ligand_preprocessor or not protein_data:
            raise ValueError("ligand_preprocessor and protein_data must be provided")
        
        # Validate SMILES in advance
        if not is_valid_smiles(str(ligand_smiles) if pd.notna(ligand_smiles) else ''):
            logger.error(f"ID {id or 'Unknown'} - Invalid SMILES: {ligand_smiles}")
            return None

        # Process ligand SMILES
        ligand_data = ligand_preprocessor.process(str(ligand_smiles) if pd.notna(ligand_smiles) else '', halogen_detail=False)
        if not ligand_data:
            logger.warning(f"ID {id or 'Unknown'} ligand processing failed, skipping")
            return None

        standardized_smiles = ligand_data['seq']
        logger.info(f"ID {id or 'Unknown'} - Original SMILES: {ligand_smiles} -> Standardized: {standardized_smiles}")

        mol = Chem.MolFromSmiles(standardized_smiles)
        if mol is None:
            logger.error(f"ID {id or 'Unknown'} - Failed to parse SMILES: {standardized_smiles}")
            return None

        # Record label information (even if there are no labels, continue processing)
        has_regression = pd.notna(regression_label)
        has_classification = pd.notna(classification_label)
        logger.info(f"ID {id or 'Unknown'} - Regression Label: {regression_label if has_regression else 'N/A'}, Classification Label: {classification_label if has_classification else 'N/A'}")

        # Build ProteinLigandData object
        data_kwargs = {
            "id": id if id else None,
            "regression_label": torch.tensor(regression_label, dtype=torch.float) if has_regression else None,
            "classification_label": torch.tensor(int(classification_label), dtype=torch.long) if has_classification else None,
            # Ligand
            "mol_x": ligand_data['mol_x'],
            "smiles": standardized_smiles,
            "atom_types": '|'.join([atom.GetSymbol() for atom in mol.GetAtoms()]),
            "chemBERTa": ligand_data['chemBERTa'],
            "mol_num_nodes": ligand_data['seq_len'],
            "mol_x_feat": ligand_data['atom_feature'],
            "mol_atom_idx": ligand_data['atom_idx'],
            "mol_edge_index": ligand_data['bond_feature'].nonzero(as_tuple=False).t().contiguous(),
            "mol_edge_attr": ligand_data['bond_feature'][ligand_data['bond_feature'].nonzero(as_tuple=False).t()[0],
                                                         ligand_data['bond_feature'].nonzero(as_tuple=False).t()[1]].long(),
            # Ligand decomposition (clique)
            "clique_x": ligand_data['x_clique'].long().view(-1, 1),
            "clique_edge_index": ligand_data['tree_edge_index'].long(),
            "clique_edge_weight": ligand_data['tree_edge_weight'].long(),
            "atom2clique_index": ligand_data['atom2clique_index'].long(),
            "clique_num_nodes": ligand_data['num_cliques'],
            # Protein
            "prot_amino_acid_sequence": amino_acid_sequence,
            "prot_node_aa": torch.from_numpy(np.array(protein_data['seq_feat'])).float(),
            "prot_node_evo": torch.from_numpy(np.array(protein_data['token_representation'])).float(),
            "prot_node_pos": torch.arange(len(protein_data['seq'])).reshape(-1, 1),
            "prot_edge_index": protein_data['edge_index'],
            "prot_edge_weight": torch.from_numpy(np.array(protein_data['edge_weight'])).float(),
            "prot_num_nodes": len(protein_data['seq']),
            "prot_x_protein": torch.from_numpy(np.array(protein_data['X_protein'])).float(),
            "prot_one_hot": torch.from_numpy(np.array(protein_data['aa_feat_oh'])).float(),
        }

        data_kwargs = {k: v for k, v in data_kwargs.items() if v is not None}
        data = ProteinLigandData(**data_kwargs)
        return data
    except Exception as e:
        logger.error(f"PDB {id or 'Unknown'} processing failed: {e}")
        return None

# Main function
if __name__ == "__main__":
    logger.info("Starting to build the dataset for prediction...")

    # Parse index file
    index_data = parse_index_file(index_file_path)
    if not index_data:
        logger.error("Failed to read index file, but will continue to attempt processing valid data.")
        index_data = []  # Prevent exit, allow manual inspection

    # Initialize ligand preprocessor
    ligand_model_path = "/chemberta_model/"
    ligand_preprocessor = JTVAEDataPreprocessor(model_path=ligand_model_path)

    # Process each PDB
    dataset = []
    for row in tqdm(index_data, desc="Processing PDB files", dynamic_ncols=True):
        # Dynamically unpack to match the tuple structure returned by parse_index_file
        pdb_id = row[0] if len(row) > 0 else None
        regression_label = row[1] if len(row) > 1 else np.nan
        amino_acid_sequence = row[2] if len(row) > 2 else None
        classification_label = row[3] if len(row) > 3 else np.nan
        ligand_smiles = row[4] if len(row) > 4 else None

        # Check necessary fields: amino_acid_sequence and ligand_smiles
        if not amino_acid_sequence or not ligand_smiles:
            logger.warning(f"PDB {pdb_id or 'Unknown'} missing necessary fields (Amino_Acid_Sequence or ligand_smiles), skipping")
            continue

        # Validate amino_acid_sequence is a string
        if not isinstance(amino_acid_sequence, str):
            logger.warning(f"PDB {pdb_id or 'Unknown'} amino_acid_sequence is not a string: {amino_acid_sequence}, skipping")
            continue

        # No longer require regression_label or classification_label
        # Record label status (only for logging)
        logger.info(f"PDB {pdb_id or 'Unknown'} - Has Regression Label: {pd.notna(regression_label)}, Has Classification Label: {pd.notna(classification_label)}")

        # Clean protein sequence
        amino_acid_sequence = clean_protein_sequence(amino_acid_sequence)
        if not amino_acid_sequence:
            logger.warning(f"PDB {pdb_id or 'Unknown'} amino_acid_sequence is empty after cleaning, skipping")
            continue

        # Generate protein features
        protein_data = process_protein_sequence(amino_acid_sequence)
        if protein_data is None:
            logger.warning(f"PDB {pdb_id or 'Unknown'} protein sequence processing failed, skipping")
            continue

        # Process PDB
        result = process_single_pdb(amino_acid_sequence, ligand_smiles, pdb_id, regression_label, classification_label, ligand_preprocessor, protein_data)
        if result:
            dataset.append(result)
        else:
            logger.warning(f"PDB {pdb_id or 'Unknown'} processing failed, skipping")

    if not dataset:
        logger.error("Dataset is empty, please check the input files or processing logic!")
    else:
        # Save dataset
        save_to_pickle(dataset, output_dir)

        # Package as PyG dataset
        pyg_dataset = ProteinLigandDataset(dataset)
        dataloader = DataLoader(pyg_dataset, batch_size=32, shuffle=False)  # Typically do not shuffle data for prediction
        logger.info(f"Dataset for prediction is complete, successfully processed {len(dataset)} PDBs.")