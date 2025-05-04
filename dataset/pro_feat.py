import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.utils import coalesce, remove_self_loops, add_self_loops, to_undirected
import esm
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file_path = "/protein_errors.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Data normalization function
def dic_normalize(dic):
    max_value = max(dic.values())
    min_value = min(dic.values())
    interval = float(max_value) - float(min_value)
    return {k: (v - min_value) / interval for k, v in dic.items()}

# Amino acid property tables
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18, 'X': 0.0}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32, 'X': 0.0}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96, 'X': 0.0}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63, 'X': 0.0}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

# Amino acid feature table (aa_features)
aa_features = {
    'A': [0.008, 0.134, -0.475, -0.039, 0.181], 'R': [0.171, -0.361, 0.107, -0.258, -0.364],
    'N': [0.255, 0.038, 0.117, 0.118, -0.055], 'D': [0.303, -0.057, -0.014, 0.225, -0.156],
    'C': [-0.132, 0.174, 0.070, -0.565, -0.374], 'Q': [0.149, -0.184, -0.030, 0.035, -0.112],
    'E': [0.221, -0.280, -0.315, 0.157, 0.303], 'G': [0.218, 0.562, -0.024, 0.018, 0.106],
    'H': [0.023, -0.177, 0.041, 0.280, -0.021], 'I': [-0.353, 0.071, -0.088, -0.195, -0.107],
    'L': [-0.267, 0.018, -0.265, -0.274, 0.206], 'K': [0.243, -0.339, -0.044, -0.325, -0.027],
    'M': [-0.239, -0.141, -0.155, 0.321, 0.077], 'F': [-0.329, -0.023, 0.072, -0.002, 0.208],
    'P': [0.173, 0.286, 0.407, -0.215, 0.384], 'S': [0.199, 0.238, -0.015, -0.068, -0.196],
    'T': [0.068, 0.147, -0.015, -0.132, -0.274], 'W': [-0.296, -0.186, 0.389, 0.083, 0.297],
    'Y': [-0.141, -0.057, 0.425, -0.096, -0.091], 'V': [0.274, 0.136, -0.187, -0.196, -0.299],
    'X': [0.0, 0.0, 0.0, 0.0, 0.0]
}

# Amino acid feature extraction function
def residue_features(residue):
    return np.array([
        1 if residue in pro_res_aliphatic_table else 0,
        1 if residue in pro_res_aromatic_table else 0,
        1 if residue in pro_res_polar_neutral_table else 0,
        1 if residue in pro_res_acidic_charged_table else 0,
        1 if residue in pro_res_basic_charged_table else 0,
        res_weight_table[residue],
        res_pka_table[residue],
        res_pl_table[residue],
        res_hydrophobic_ph7_table[residue]
    ])

# Split sequence features
def seq_feature(sequence):
    seq_len = len(sequence)
    seq_feat = np.zeros((seq_len, 9))
    for i, aa in enumerate(sequence):
        if aa not in res_weight_table:
            aa = 'X'
        seq_feat[i] = residue_features(aa)
    return seq_feat

# One-hot encoding of amino acids
def one_hot_encoding(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
    one_hot = {aa: [1 if aa == a else 0 for a in amino_acids] for aa in amino_acids}
    seq_len = len(sequence)
    aa_feat_oh = np.zeros((seq_len, len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa not in amino_acids:
            aa = 'X'
        aa_feat_oh[i] = one_hot[aa]
    return aa_feat_oh

# ESM feature extraction function (PCA dimensionality reduction removed)
def esm_extract(model, batch_converter, sequence, layer=33, approach='mean', dim=1280, chunk_size=350):
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device
    seq_len = len(sequence)
    token_representation = np.zeros((seq_len, dim))
    contact_map = np.zeros((seq_len, seq_len))

    for start in tqdm(range(0, seq_len, chunk_size), desc="Processing sequence chunks"):
        end = min(start + chunk_size * 2, seq_len)
        sub_seq = sequence[start:end]
        
        data = [("protein", sub_seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer], return_contacts=True)

        sub_token_repr = results["representations"][layer][0, 1:len(sub_seq)+1].cpu().numpy()
        overlap = min(end - seq_len, chunk_size)
        
        token_representation[start:end] += sub_token_repr
        if overlap > 0:
            token_representation[start+chunk_size:end] /= 2  # Average overlapping parts

        sub_contact_map = results["contacts"][0].cpu().numpy()
        contact_map[start:end, start:end] += sub_contact_map
        if overlap > 0:
            contact_map[start+chunk_size:end, start+chunk_size:end] /= 2  # Average overlapping parts

    # Remove PCA dimensionality reduction, return original features directly
    return token_representation, contact_map

# Graph structure construction
def contact_map(contact_map_proba, contact_threshold=0.5):
    num_residues = contact_map_proba.shape[0]
    # Convert NumPy array to PyTorch tensor
    contact_map_proba = torch.from_numpy(contact_map_proba)
    prot_contact_adj = (contact_map_proba >= contact_threshold).long()
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
    logger.info(f"Initial shape of edge_index: {edge_index.shape}")  # Debugging
    row = edge_index[0]
    col = edge_index[1]
    edge_weight = contact_map_proba[row, col].float()

    # Connect isolated nodes
    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1], (torch.arange(num_residues)+1)[:-1]])
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:], (torch.arange(num_residues)-1)[1:]])
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    logger.info(f"Shape of seq_edge_head1: {seq_edge_head1.shape}, shape of seq_edge_tail1: {seq_edge_tail1.shape}")  # Debugging
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1], dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2], (torch.arange(num_residues)+2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:], (torch.arange(num_residues)-2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2], dim=-1)

    # Graph operations
    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1)
    
    logger.info(f"Final shape of edge_index: {edge_index.shape}, shape of edge_weight: {edge_weight.shape}")  # Debugging
    return edge_index, edge_weight

# Process multiple protein sequences
def process_protein_sequence(sequence, model_path="/esm2_t33_650M_UR50D.pt", 
                             approach='mean'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Validate that the input sequence is a string
    if not isinstance(sequence, str):
        logger.error(f"Input sequence is not a string: {sequence}, type: {type(sequence)}")
        return None
    
    # Validate that the sequence is not empty
    if not sequence.strip():
        logger.error(f"Input sequence is empty or contains only whitespace characters: {sequence}")
        return None

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    try:
        logger.info(f"Processing sequence: {sequence[:10]}...")
        # Extract sequence features
        X_protein = np.array([[i] + aa_features.get(aa, aa_features['X']) for i, aa in enumerate(sequence)])
        seq_feat = seq_feature(sequence)
        aa_feat_oh = one_hot_encoding(sequence)
        
        # Extract ESM features (without PCA)
        token_representation, contact_map_proba = esm_extract(model, batch_converter, sequence, approach=approach)
        
        # Construct graph structure
        edge_index, edge_weight = contact_map(contact_map_proba)

        # Print the shape of each feature
        logger.info(f"Sequence: {sequence[:10]}...")
        logger.info(f"seq_len: {len(sequence)}")
        logger.info(f"Shape of X_protein: {X_protein.shape}")  # [seq_len, 6]
        logger.info(f"Shape of seq_feat: {seq_feat.shape}")    # [seq_len, 9]
        logger.info(f"Shape of aa_feat_oh: {aa_feat_oh.shape}")  # [seq_len, 21]
        logger.info(f"Shape of token_representation: {token_representation.shape}")  # [seq_len, 1280]
        logger.info(f"Shape of edge_index: {edge_index.shape}")  # [2, num_edges]
        logger.info(f"Shape of edge_weight: {edge_weight.shape}")  # [num_edges]

        # Return result dictionary
        return {
            "seq": sequence,  # Protein sequence
            "seq_len": len(sequence),  # Sequence length
            "X_protein": X_protein,  # Feature vector for each amino acid (including index and aa_features)
            "seq_feat": seq_feat,  # Sequence features (based on amino acid classification and physicochemical properties)
            "aa_feat_oh": aa_feat_oh,  # One-hot encoding features of amino acids
            "token_representation": token_representation,  # Original feature representation extracted by ESM model
            "edge_index": edge_index,  # Edge indices of the graph structure
            "edge_weight": edge_weight  # Edge weights of the graph structure
        }
    except Exception as e:
        import traceback
        logger.error(f"Error occurred while processing protein sequence: {e}\n{traceback.format_exc()}. Sequence: {sequence}")
        return None

# Example main function
def main():
    # Example sequence
    sequence = "ACDEFGHIKLMNPQRSTVWYX"  # Replace with your actual sequence
    model_path = "esm2_t33_650M_UR50D.pt"
    
    seq_feat = seq_feature(sequence)
    print("Shape of seq_feat:", seq_feat.shape)  # Should output (21, 9)
    
    try:
        result = process_protein_sequence(sequence, model_path=model_path)
        print("Processing complete, result:", result)
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()