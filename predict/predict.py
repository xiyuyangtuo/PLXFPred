import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
import pickle
from data_utils import ProteinLigandDataset, setup_logging, custom_collate_fn
from model import FusedAffinityPredictor
from visualize_attention import visualize_all_attentions  
from molecule_graph import MoleculeGraph
from dataset import ProteinLigandData
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from pro_feat import process_protein_sequence
import networkx as nx
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import pandas as pd
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize logging
log_dir = "/logs/prediction1"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "prediction_log.txt")
setup_logging(log_path)
logging.info("Starting prediction and visualization process")

# Load the test dataset
test_dataset = ProteinLigandDataset(
    pkl_path="predict_dataset_full.pkl",
    normalize_affinity=True,
    training=False
)
logging.info(f"Test dataset loaded with {len(test_dataset)} samples")

# Create DataLoader
batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Initialize the model
model = FusedAffinityPredictor(
    atom_feat_dim=43,
    mol_edge_attr_dim=1,
    hidden_dim=128,
    chemberta_dim=768,
    gat_heads_protein=2,
    gat_heads_ligand=2,
    transformer_nhead=2,
    cross_heads=8,
    input_dropout=0.22129538276104416,
    hidden_dropout=0.2
).to(device)

# Load the trained model weights
model_path = "best_model_multi_task1.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"Model weights loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"Failed to load model weights: {e}")
    raise

model.eval()

# Atomic number to symbol mapping
ATOM_NUM_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl',
    35: 'Br', 53: 'I'
}

# Define a custom colormap: purple to red
colors = [
    '#440154',  # Dark purple (same as viridis start)
    '#3B528B',  # Intermediate purple-blue
    '#982D80',  # Purple-red transition
    '#FF0000'   # Bright red
]
custom_cmap = LinearSegmentedColormap.from_list('purple_to_red', colors)

# Visualize the ligand molecule graph (edge ​​weights)
def visualize_molecular_graph_with_weights(smiles, bond_weights, mol_edge_index, output_path, sample_id):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        Chem.rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
        drawer.drawOptions().addAtomIndices = False

        bond_weight_dict = {}
        for bond in mol.GetBonds():
            src = bond.GetBeginAtomIdx()
            dst = bond.GetEndAtomIdx()
            weight = bond_weights[src, dst] if bond_weights[src, dst] > 0 else bond.GetBondTypeAsDouble()
            bond_weight_dict[bond.GetIdx()] = f"{weight:.2f}"

        for bond_idx, weight in bond_weight_dict.items():
            mol.GetBondWithIdx(bond_idx).SetProp('bondNote', weight)

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        with open(output_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        logging.info(f"Sample {sample_id}: Molecular graph with weights saved to {output_path}")
    except Exception as e:
        logging.error(f"Sample {sample_id}: Failed to visualize molecular graph: {e}")

# Visualize ligand molecule graph (node ​​weights)
def visualize_molecular_graph_with_node_weights(smiles, node_weights, mol_edge_index, output_path, sample_id):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        Chem.rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(800, 800)
        drawer.drawOptions().addAtomIndices = True

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx in node_weights:
                weight = node_weights[idx]
                if isinstance(weight, (list, np.ndarray)):
                    weight = np.mean(weight)
                atom.SetProp('atomNote', f"{weight:.2f}")

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        with open(output_path, 'wb') as f:
            f.write(drawer.GetDrawingText())
        logging.info(f"Sample {sample_id}: Molecular graph with node weights saved to {output_path}")
    except Exception as e:
        logging.error(f"Sample {sample_id}: Failed to visualize molecular graph with node weights: {e}")


# Visualizing protein contact graphs (node ​​weights)
def visualize_protein_contact_with_node_weights(edge_index, edge_weight, node_weights, sequence, save_path):
    try:
        seq_len = len(sequence)
        logging.info(f"Sample: Visualizing node weights map for sequence length: {seq_len}")
        logging.info(f"Sample: Edge index shape: {edge_index.shape}, Edge weight shape: {edge_weight.shape}")
        logging.info(f"Sample: Node weights keys: {list(node_weights.keys())}")

        contact_matrix = np.zeros((seq_len, seq_len))
        for idx, (src, dst) in enumerate(edge_index.T):
            src, dst = int(src), int(dst)
            if src < seq_len and dst < seq_len:
                contact_prob = edge_weight[idx] if idx < len(edge_weight) else 0.0
                contact_matrix[src, dst] = contact_prob
                contact_matrix[dst, src] = contact_prob
            else:
                logging.warning(f"Sample: Invalid edge indices: src={src}, dst={dst} (seq_len={seq_len})")

        logging.info(f"Sample: Contact matrix max: {np.max(contact_matrix)}, min: {np.min(contact_matrix)}")

        fig, ax = plt.subplots(figsize=(10, 8))
        contact_norm = Normalize(vmin=0, vmax=max(np.max(contact_matrix), 1e-6))
        cmap_contact = plt.get_cmap('viridis')
        sns.heatmap(contact_matrix, cmap=cmap_contact, norm=contact_norm, square=True, cbar=False, ax=ax)

        avg_node_weights = {node: np.mean(weights) for node, weights in node_weights.items()}
        if not avg_node_weights:
            logging.warning("Sample: No node weights available, using zeros")
            avg_node_weights = {i: 0.0 for i in range(seq_len)}

        node_weight_norm = Normalize(vmin=min(avg_node_weights.values()), vmax=max(avg_node_weights.values(), default=1e-6))
        cmap_node = custom_cmap
        for i in range(seq_len):
            if i in avg_node_weights:
                alpha = node_weight_norm(avg_node_weights[i])
                ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=True, color=cmap_node(alpha), alpha=0.5, zorder=2))
                logging.info(f"Sample: Node {i} weight: {avg_node_weights[i]:.4f}, alpha: {alpha:.4f}")

        tick_step = max(1, seq_len // 10)
        ax.set_xticks(np.arange(0, seq_len, tick_step))
        ax.set_yticks(np.arange(0, seq_len, tick_step))
        ax.set_xticklabels(np.arange(0, seq_len, tick_step), rotation=0)
        ax.set_yticklabels(np.arange(0, seq_len, tick_step), rotation=0)
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Residue Index')
        ax.set_title('Protein Contact Map with Node Attention Weights')

        cbar_contact = fig.colorbar(ScalarMappable(norm=contact_norm, cmap=cmap_contact), ax=ax, label='Contact Probability')
        cbar_node = fig.colorbar(ScalarMappable(norm=node_weight_norm, cmap=cmap_node), ax=ax, label='Node Attention Weight (Purple to Red)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Sample: Protein contact map with node weights saved to {save_path}")
    except Exception as e:
        logging.error(f"Sample: Failed to visualize protein contact map with node weights: {e}")

# Visualizing protein contact graph
def visualize_protein_contact_with_edge_weights(edge_index, edge_weights, edge_contact_weights, sequence, save_path):
    try:
        seq_len = len(sequence)
        logging.info(f"Sample: Visualizing edge weights map for sequence length: {seq_len}")
        logging.info(f"Sample: Edge index shape: {edge_index.shape}, Edge weights shape: {edge_weights.shape}, Contact weights shape: {edge_contact_weights.shape}")

        contact_matrix = np.zeros((seq_len, seq_len))
        attention_matrix = np.zeros((seq_len, seq_len))
        
        for idx, (src, dst) in enumerate(edge_index.T):
            src, dst = int(src), int(dst)
            if src < seq_len and dst < seq_len:
                contact_prob = edge_contact_weights[idx] if idx < len(edge_contact_weights) else 0.0
                contact_matrix[src, dst] = contact_prob
                contact_matrix[dst, src] = contact_prob
            else:
                logging.warning(f"Sample: Invalid edge indices: src={src}, dst={dst} (seq_len={seq_len})")

        avg_edge_weights = np.mean(edge_weights, axis=1)
        for idx, (src, dst) in enumerate(edge_index.T):
            src, dst = int(src), int(dst)
            if src < seq_len and dst < seq_len:
                weight = avg_edge_weights[idx] if idx < len(avg_edge_weights) else 0.0
                attention_matrix[src, dst] = weight
                attention_matrix[dst, src] = weight
                logging.info(f"Sample: Edge {src}-{dst} weight: {weight:.4f}")

        logging.info(f"Sample: Contact matrix max: {np.max(contact_matrix)}, min: {np.min(contact_matrix)}")
        logging.info(f"Sample: Attention matrix max: {np.max(attention_matrix)}, min: {np.min(attention_matrix)}")

        fig, ax = plt.subplots(figsize=(10, 8))
        contact_norm = Normalize(vmin=0, vmax=max(np.max(contact_matrix), 1e-6))
        cmap_contact = plt.get_cmap('viridis')
        sns.heatmap(contact_matrix, cmap=cmap_contact, norm=contact_norm, square=True, cbar=False, ax=ax)

        edge_weight_norm = Normalize(vmin=0, vmax=max(np.max(attention_matrix), 1e-6))
        cmap_edge = custom_cmap
        for i in range(seq_len):
            for j in range(seq_len):
                if attention_matrix[i, j] > 0:
                    alpha = edge_weight_norm(attention_matrix[i, j])
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=cmap_edge(alpha), alpha=0.5, zorder=2))

        tick_step = max(1, seq_len // 10)
        ax.set_xticks(np.arange(0, seq_len, tick_step))
        ax.set_yticks(np.arange(0, seq_len, tick_step))
        ax.set_xticklabels(np.arange(0, seq_len, tick_step), rotation=0)
        ax.set_yticklabels(np.arange(0, seq_len, tick_step), rotation=0)
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Residue Index')
        ax.set_title('Protein Contact Map with Edge Attention Weights')

        cbar_contact = fig.colorbar(ScalarMappable(norm=contact_norm, cmap=cmap_contact), ax=ax, label='Contact Probability')
        cbar_edge = fig.colorbar(ScalarMappable(norm=edge_weight_norm, cmap=cmap_edge), ax=ax, label='Edge Attention Weight (Purple to Red)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Sample: Protein contact map with edge weights saved to {save_path}")
    except Exception as e:
        logging.error(f"Sample: Failed to visualize protein contact map with edge weights: {e}")

# Modified log_attention_on_graphs
def log_attention_on_graphs(model, batch, log_dir, sample_id, attention_threshold=0.1):
    try:
        logging.info(f"Sample {sample_id}: Batch attributes: {list(batch.keys())}")

        # --- Ligand Graph (Molecule) ---
        possible_smiles_fields = ['smiles', 'ligand_smiles', 'mol_smiles', 'smiles_string', 'SMILES']
        smiles = None
        for field in possible_smiles_fields:
            if field in batch:
                smiles = batch[field]
                if isinstance(smiles, list):
                    smiles = smiles[0]
                logging.info(f"Sample {sample_id}: Found SMILES in field '{field}': {smiles}")
                break

        if smiles:
            mol_graph = MoleculeGraph(halogen_detail=False)
            graph_data = mol_graph.smiles2graph(smiles)
            atom_types = graph_data['atom_types'].split('|')
            bond_feature = graph_data['bond_feature'].numpy()
            bond_weights = graph_data['bond_weights'].numpy()
        else:
            logging.warning(f"Sample {sample_id}: SMILES field not found. Falling back to smiles_atomic and mol_edge_index.")
            if len(batch.smiles_atomic.shape) == 1:
                atom_types = [ATOM_NUM_TO_SYMBOL.get(atom.item(), f"Unknown({atom.item()})") for atom in batch.smiles_atomic]
            elif len(batch.smiles_atomic.shape) == 2:
                atom_types = [ATOM_NUM_TO_SYMBOL.get(atom[0].item(), f"Unknown({atom[0].item()})") for atom in batch.smiles_atomic]
            else:
                raise ValueError(f"Unsupported smiles_atomic shape: {batch.smiles_atomic.shape}")
            bond_feature = np.zeros((len(atom_types), len(atom_types)))
            bond_weights = np.zeros((len(atom_types), len(atom_types)))

        mol_edge_index = batch.mol_edge_index.numpy()
        mol_attn_weights = model.lig_graph_enc.gat_mol_attn_weights.cpu().numpy()
        mol_avg_attn_weights = mol_attn_weights.mean(axis=1)

        logging.info(f"Sample {sample_id}: Ligand edge attention weights:")
        for idx, (src, dst) in enumerate(mol_edge_index.T):
            weight = mol_avg_attn_weights[idx] if idx < len(mol_avg_attn_weights) else 0
            src_symbol = atom_types[src] if src < len(atom_types) else f"Unknown({src})"
            dst_symbol = atom_types[dst] if dst < len(atom_types) else f"Unknown({dst})"
            logging.info(f"Edge: {src_symbol}{src}-{dst_symbol}{dst}, Attention Weight: {weight:.4f}")

        node_attn_weights = defaultdict(float)
        node_edge_count = defaultdict(int)
        for idx, (src, dst) in enumerate(mol_edge_index.T):
            weight = mol_avg_attn_weights[idx] if idx < len(mol_avg_attn_weights) else 0
            node_attn_weights[src] += weight
            node_attn_weights[dst] += weight
            node_edge_count[src] += 1
            node_edge_count[dst] += 1
        mol_node_weights = {node: node_attn_weights[node] / max(node_edge_count[node], 1) 
                           for node in node_attn_weights}

        logging.info(f"Sample {sample_id}: Ligand node attention weights:")
        for node_idx, weight in sorted(mol_node_weights.items()):
            atom_symbol = atom_types[node_idx] if node_idx < len(atom_types) else f"Unknown({node_idx})"
            logging.info(f"Node: {atom_symbol}{node_idx}, Attention Weight: {weight:.4f}")

        if smiles:
            vis_dir = os.path.join(log_dir, f"sample_{sample_id}")
            os.makedirs(vis_dir, exist_ok=True)
            vis_path_bonds = os.path.join(vis_dir, "molecular_graph_with_weights.png")
            visualize_molecular_graph_with_weights(smiles, bond_weights, mol_edge_index, vis_path_bonds, sample_id)
            vis_path_nodes = os.path.join(vis_dir, "molecular_graph_with_node_weights.png")
            visualize_molecular_graph_with_node_weights(smiles, mol_node_weights, mol_edge_index, vis_path_nodes, sample_id)

        # --- Protein Graph ---
        possible_sequence_fields = ['prot_amino_acid_sequence', 'sequence', 'Protein sequence', 'seq', 'prot_seq']
        sequence = None
        for field in possible_sequence_fields:
            if field in batch:
                sequence = batch[field]
                if isinstance(sequence, list):
                    sequence = sequence[0]
                logging.info(f"Sample {sample_id}: Found protein sequence in field '{field}': {sequence[:10]}... (length: {len(sequence)})")
                break

        if sequence:
            protein_data = process_protein_sequence(sequence)
            if protein_data is None:
                logging.error(f"Sample {sample_id}: Failed to process protein sequence.")
            else:
                logging.info(f"Sample {sample_id}: Protein data processed. Edge index shape: {protein_data['edge_index'].shape}, Edge weight shape: {protein_data['edge_weight'].shape}")
                batch.prot_edge_index = torch.tensor(protein_data['edge_index'], dtype=torch.long)
                batch.prot_edge_weight = torch.tensor(protein_data['edge_weight'], dtype=torch.float)
                batch.prot_node_evo = torch.randn(len(sequence), 1280)
                batch.prot_one_hot = torch.randn(len(sequence), 21)
                batch.prot_node_aa = torch.randn(len(sequence), 9)
                batch.prot_node_pos = torch.arange(len(sequence)).unsqueeze(-1)
                batch.prot_batch = torch.zeros(len(sequence), dtype=torch.long)

                batch.prot_edge_index = batch.prot_edge_index.to('cpu')
                batch.prot_edge_weight = batch.prot_edge_weight.to('cpu')

                with torch.no_grad():
                    model.prot_graph_enc(batch)

                prot_edge_index = protein_data['edge_index']
                prot_edge_weight = protein_data['edge_weight']
                prot_attn_weights = model.prot_graph_enc.gat2_attn_weights.cpu().numpy()  # Shape: (num_edges, num_heads)

                # Log edge attention weights for each head
                edge_dict = defaultdict(list)
                num_heads = prot_attn_weights.shape[1]  # Number of attention heads
                for idx, (src, dst) in enumerate(prot_edge_index.T):
                    weights = prot_attn_weights[idx]  # Shape: (num_heads,)
                    edge_key = tuple(sorted([int(src), int(dst)]))
                    edge_dict[edge_key].append(weights)

                logging.info(f"Sample {sample_id}: Protein edge attention weights (undirected, per head):")
                for (src, dst), weights_list in edge_dict.items():
                    avg_weights = np.mean(weights_list, axis=0)  # Shape: (num_heads,)
                    if np.all(avg_weights < attention_threshold):
                        continue
                    src_aa = sequence[src] if src < len(sequence) else 'X'
                    dst_aa = sequence[dst] if dst < len(sequence) else 'X'
                    weights_str = ", ".join([f"Head {h}: {w:.4f}" for h, w in enumerate(avg_weights)])
                    logging.info(f"Edge: {src_aa}{src}-{dst_aa}{dst}, Attention Weights: {weights_str}")

                # Compute node attention weights per head
                prot_node_attn_weights = defaultdict(lambda: np.zeros(num_heads))
                prot_node_edge_count = defaultdict(int)
                for (src, dst), weights_list in edge_dict.items():
                    avg_weights = np.mean(weights_list, axis=0)  # Shape: (num_heads,)
                    if np.all(avg_weights < attention_threshold):
                        continue
                    prot_node_attn_weights[src] += avg_weights
                    prot_node_attn_weights[dst] += avg_weights
                    prot_node_edge_count[src] += 1
                    prot_node_edge_count[dst] += 1

                # Normalize node weights
                prot_node_weights = {
                    node: prot_node_attn_weights[node] / max(prot_node_edge_count[node], 1)
                    for node in prot_node_attn_weights
                }

                logging.info(f"Sample {sample_id}: Protein node attention weights (per head):")
                for node_idx, weights in sorted(prot_node_weights.items()):
                    aa = sequence[node_idx] if node_idx < len(sequence) else 'X'
                    weights_str = ", ".join([f"Head {h}: {w:.4f}" for h, w in enumerate(weights)])
                    logging.info(f"Node: {aa}{node_idx}, Attention Weights: {weights_str}")

                vis_dir = os.path.join(log_dir, f"sample_{sample_id}")
                os.makedirs(vis_dir, exist_ok=True)
                logging.info(f"Sample {sample_id}: Visualization directory: {vis_dir}")
                
                vis_path_nodes = os.path.join(vis_dir, "protein_contact_map_with_node_weights.png")
                logging.info(f"Sample {sample_id}: Attempting to save node weights map to {vis_path_nodes}")
                visualize_protein_contact_with_node_weights(prot_edge_index, prot_edge_weight, prot_node_weights, sequence, vis_path_nodes)
                
                vis_path_edges = os.path.join(vis_dir, "protein_contact_map_with_edge_weights.png")
                logging.info(f"Sample {sample_id}: Attempting to save edge weights map to {vis_path_edges}")
                visualize_protein_contact_with_edge_weights(prot_edge_index, prot_attn_weights, prot_edge_weight, sequence, vis_path_edges)

        else:
            logging.warning(f"Sample {sample_id}: No protein sequence found in batch.")

    except Exception as e:
        logging.error(f"Sample {sample_id}: Error in logging attention on graphs: {e}")

# Prediction and visualization functions
def predict_and_visualize(model, loader, device, log_dir, classification_threshold=0.1, 
                          regression_mean=5.0, regression_std=2.0):
    predictions_reg = []
    predictions_cls = []
    sample_ids = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            try:
                batch = batch.to(device)
                logging.info(f"Processing batch {i + 1}/{len(loader)}")

                # Extract the sample ID from batch.id
                if hasattr(batch, 'id'):
                    sample_id = batch.id
                    if isinstance(sample_id, torch.Tensor):
                        sample_id = sample_id.cpu().numpy().flatten()[0]
                    elif isinstance(sample_id, list):
                        sample_id = sample_id[0]
                    sample_id = str(sample_id)  # Ensure it's a string for consistency
                else:
                    logging.warning(f"Batch {i}: No 'id' field found in batch. Using default ID 'sample_{i}'.")
                    sample_id = f"sample_{i}"
                sample_ids.append(sample_id)

                reg_output, cls_output = model(batch)
                
                current_reg_pred = reg_output.cpu().numpy().flatten()
                current_reg_pred_denorm = current_reg_pred * regression_std + regression_mean
                predictions_reg.extend(current_reg_pred_denorm)
                logging.info(f"Sample {sample_id}: Regression prediction (denormalized) = {current_reg_pred_denorm}")

                current_cls_pred = torch.sigmoid(cls_output).cpu().numpy().flatten()
                current_cls_pred_calibrated = 1 / (1 + np.exp(-10 * (current_cls_pred - 0.05)))
                predictions_cls.extend(current_cls_pred_calibrated)
                logging.info(f"Sample {sample_id}: Classification prediction (calibrated) = {current_cls_pred_calibrated}")

                model.to('cpu')
                batch = batch.to('cpu')
                vis_dir = os.path.join(log_dir, f"sample_{sample_id}")
                os.makedirs(vis_dir, exist_ok=True)
                
                try:
                    vis_path = os.path.join(vis_dir, "attention_visualization.png")
                    visualize_all_attentions(model, batch, vis_path)
                    logging.info(f"General attention visualization saved to {vis_path}")
                except Exception as e:
                    logging.error(f"Failed to visualize attention for sample {sample_id}: {e}")
                
                try:
                    log_attention_on_graphs(model, batch, log_dir, sample_id)
                except Exception as e:
                    logging.error(f"Failed to analyze attention for sample {sample_id}: {e}")
                
                model.to(device)
                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Error processing batch {i}: {e}")
                continue

    logging.info(f"Classification probability stats: min={np.min(predictions_cls):.4f}, max={np.max(predictions_cls):.4f}, mean={np.mean(predictions_cls):.4f}, median={np.median(predictions_cls):.4f}")

    results_dict = {
        'sample_id': sample_ids,
        'regression_prediction': predictions_reg,
        'classification_prediction': predictions_cls,
        'classification_pred_class': (np.array(predictions_cls) > classification_threshold).astype(int)
    }

    results_df = pd.DataFrame(results_dict)
    
    csv_path = os.path.join(log_dir, "predictions.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Predictions saved to CSV file: {csv_path}")
    
    results = {
        'regression_predictions': np.array(predictions_reg),
        'classification_predictions': np.array(predictions_cls),
        'classification_pred_class': (np.array(predictions_cls) > classification_threshold).astype(int),
        'classification_threshold': classification_threshold
    }
    results_path = os.path.join(log_dir, "predictions.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"Predictions also saved to pickle file: {results_path}")

    return results_df

if __name__ == "__main__":
    try:
        regression_mean = 5.0
        regression_std = 2.0
        results_df = predict_and_visualize(
            model, 
            test_loader, 
            device, 
            log_dir, 
            classification_threshold=0.1,
            regression_mean=regression_mean,
            regression_std=regression_std
        )
        logging.info("Prediction and visualization process completed successfully")
        print(results_df)
    except Exception as e:
        logging.error(f"Prediction process failed: {e}")
        raise