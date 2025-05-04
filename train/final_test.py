import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import pandas as pd
from lifelines.utils import concordance_index
import seaborn as sns
from dataset import ProteinLigandData
from model import FusedAffinityPredictor
from data_utils import ProteinLigandDataset, setup_logging, custom_collate_fn
import shap
from torch_geometric.nn import global_max_pool
from visualize_attention import (shap_summary_bar, shap_waterfall_plot, 
                                 shap_feature_impact, shap_density_scatter, 
                                 shap_hierarchical_clustering, shap_dependence_plots, 
                                 shap_heatmap, shap_sample_clustering_heatmap)

# Initialize logging
log_dir = " "
log_path = os.path.join(log_dir, ".txt")
setup_logging(log_path)
logging.info("Starting testing process with SHAP analysis")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enhanced Evaluate Model Function with AUPRC
def evaluate_model(model, loader, device, is_classification=True):
    model.eval()
    y_true_cls, y_pred_cls, y_pred_cls_prob = [], [], []
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                regression_output, classification_output = model(batch)
                if hasattr(batch, 'classification_label'):
                    y_true_cls.extend(batch.classification_label.cpu().numpy())
                    y_pred_cls_prob.extend(torch.sigmoid(classification_output).cpu().numpy().flatten())
                    y_pred_cls.extend(torch.sigmoid(classification_output).round().cpu().numpy().flatten())
        
        metrics = {}
        if y_true_cls and is_classification:
            y_true_cls = np.array(y_true_cls)
            y_pred_cls = np.array(y_pred_cls)
            y_pred_cls_prob = np.array(y_pred_cls_prob)
            valid_mask = np.isfinite(y_true_cls) & np.isfinite(y_pred_cls_prob)
            if valid_mask.sum() > 0:
                y_true_cls_clean = y_true_cls[valid_mask]
                y_pred_cls_clean = y_pred_cls[valid_mask]
                y_pred_cls_prob_clean = y_pred_cls_prob[valid_mask]
                precision, recall, _ = precision_recall_curve(y_true_cls_clean, y_pred_cls_prob_clean)
                metrics.update({
                    'accuracy': accuracy_score(y_true_cls_clean, y_pred_cls_clean),
                    'roc_auc': roc_auc_score(y_true_cls_clean, y_pred_cls_prob_clean),
                    'f1': f1_score(y_true_cls_clean, y_pred_cls_clean, average='weighted'),
                    'fpr': roc_curve(y_true_cls_clean, y_pred_cls_prob_clean)[0],
                    'tpr': roc_curve(y_true_cls_clean, y_pred_cls_prob_clean)[1],
                    'precision': precision,
                    'recall': recall,
                    'ap': average_precision_score(y_true_cls_clean, y_pred_cls_prob_clean),  # AUPRC
                    'conf_matrix': confusion_matrix(y_true_cls_clean, y_pred_cls_clean)
                })
            else:
                metrics.update({
                    'accuracy': 0.0, 'roc_auc': 0.0, 'f1': 0.0, 'fpr': np.array([]), 'tpr': np.array([]),
                    'precision': np.array([]), 'recall': np.array([]), 'ap': 0.0, 'conf_matrix': np.array([[0, 0], [0, 0]])
                })
        return metrics
    except Exception as e:
        logging.error(f"Error in evaluate_model: {e}")
        return {
            'accuracy': 0.0, 'roc_auc': 0.0, 'f1': 0.0, 'fpr': np.array([]), 'tpr': np.array([]),
            'precision': np.array([]), 'recall': np.array([]), 'ap': 0.0, 'conf_matrix': np.array([[0, 0], [0, 0]])
        }

# Visualization Functions
def plot_roc_curve(fpr, tpr, roc_auc, dataset_name, log_dir):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plot_path = os.path.join(log_dir, f'roc_curve_{dataset_name.lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"ROC curve for {dataset_name} saved to {plot_path}")

def plot_precision_recall_curve(precision, recall, ap, dataset_name, log_dir):
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUPRC = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, f'pr_curve_{dataset_name.lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Precision-Recall curve for {dataset_name} saved to {plot_path}")

def plot_confusion_matrix(metrics_dict, dataset_name, log_dir):
    conf_matrix = metrics_dict['conf_matrix']
    if conf_matrix.size > 0:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plot_path = os.path.join(log_dir, f'confusion_matrix_{dataset_name.lower()}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Confusion matrix for {dataset_name} saved to {plot_path}")

# New function to plot Accuracy and F1 as a bar chart
def plot_classification_metrics(metrics, dataset_name, log_dir):
    plt.figure(figsize=(6, 4))
    metrics_names = ['Accuracy', 'F1']
    values = [metrics['accuracy'], metrics['f1']]
    bars = plt.bar(metrics_names, values, color=['skyblue', 'lightcoral'])
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title(f'Classification Metrics - {dataset_name}')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plot_path = os.path.join(log_dir, f'classification_metrics_{dataset_name.lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Classification metrics plot for {dataset_name} saved to {plot_path}")

# Function to compute intermediate features with corrected scatter_add_
def compute_intermediate_features(batch, model, device):
    batch = batch.to(device)
    with torch.no_grad():
        # Compute raw features
        _, prot_graph_raw = model.prot_graph_enc(batch)
        _, lig_graph_raw = model.lig_graph_enc(batch)
        
        # Ensure all attributes are tensors and log their shapes
        logging.debug(f"batch.prot_batch shape: {batch.prot_batch.shape}, max: {batch.prot_batch.max()}")
        logging.debug(f"batch.clique_batch shape: {batch.clique_batch.shape}, max: {batch.clique_batch.max()}")
        logging.debug(f"batch.mol_batch shape: {batch.mol_batch.shape}, max: {batch.mol_batch.max()}")
        
        # Pool protein edge features
        num_graphs = batch.prot_batch.max().item() + 1
        prot_edge_weight_pooled = torch.zeros(num_graphs, 1, device=device, dtype=torch.float32)
        num_edges = batch.prot_edge_index.size(1)
        
        if batch.prot_edge_weight.size(0) != num_edges:
            batch.prot_edge_weight = batch.prot_edge_weight[:num_edges] if batch.prot_edge_weight.size(0) > num_edges else torch.cat([batch.prot_edge_weight, torch.ones(num_edges - batch.prot_edge_weight.size(0), device=device)])
        
        valid_prot_edge_mask = (batch.prot_edge_index[0] < num_graphs) & (batch.prot_edge_index[1] < num_graphs)
        prot_edge_index_filtered = batch.prot_edge_index[:, valid_prot_edge_mask]
        prot_edge_weight_filtered = batch.prot_edge_weight[valid_prot_edge_mask]
        
        if prot_edge_index_filtered.size(1) > 0:
            prot_indices = batch.prot_batch[prot_edge_index_filtered[0]].unsqueeze(-1)
            prot_edge_weight_pooled = prot_edge_weight_pooled.scatter_add_(0, prot_indices, prot_edge_weight_filtered.unsqueeze(-1))
            counts = torch.bincount(batch.prot_batch[prot_edge_index_filtered[0]], minlength=num_graphs).float().clamp(min=1).unsqueeze(-1)
            prot_edge_weight_pooled = prot_edge_weight_pooled / counts
        
        # Pool clique edge features
        num_graphs_clique = batch.clique_batch.max().item() + 1
        clique_edge_weight_pooled = torch.zeros(num_graphs_clique, 1, device=device, dtype=torch.float32)
        num_edges_clique = batch.clique_edge_index.size(1)
        
        if batch.clique_edge_weight.size(0) != num_edges_clique:
            batch.clique_edge_weight = batch.clique_edge_weight[:num_edges_clique] if batch.clique_edge_weight.size(0) > num_edges_clique else torch.cat([batch.clique_edge_weight, torch.ones(num_edges_clique - batch.clique_edge_weight.size(0), device=device)])
        
        valid_clique_edge_mask = (batch.clique_edge_index[0] < num_graphs_clique) & (batch.clique_edge_index[1] < num_graphs_clique)
        clique_edge_index_filtered = batch.clique_edge_index[:, valid_clique_edge_mask]
        clique_edge_weight_filtered = batch.clique_edge_weight[valid_clique_edge_mask]
        
        if clique_edge_index_filtered.size(1) > 0:
            clique_indices = batch.clique_batch[clique_edge_index_filtered[0]].unsqueeze(-1)
            clique_edge_weight_pooled = clique_edge_weight_pooled.scatter_add_(0, clique_indices, clique_edge_weight_filtered.unsqueeze(-1))
            counts_clique = torch.bincount(batch.clique_batch[clique_edge_index_filtered[0]], minlength=num_graphs_clique).float().clamp(min=1).unsqueeze(-1)
            clique_edge_weight_pooled = clique_edge_weight_pooled / counts_clique
        
        # Pool node features
        prot_node_evo_pooled = global_max_pool(batch.prot_node_evo, batch.prot_batch)
        prot_node_aa_pooled = global_max_pool(batch.prot_node_aa, batch.prot_batch)
        mol_x_feat_pooled = global_max_pool(batch.mol_x_feat, batch.mol_batch)
        clique_x_pooled = global_max_pool(batch.clique_x, batch.clique_batch)
        
        # Pool molecular edge features
        num_nodes_mol = batch.mol_batch.size(0)
        mol_edge_attr_expanded = batch.mol_edge_attr.unsqueeze(-1)
        valid_mol_edge_mask = (batch.mol_edge_index[1] < num_nodes_mol)
        mol_edge_index_filtered = batch.mol_edge_index[:, valid_mol_edge_mask]
        mol_edge_attr_filtered = mol_edge_attr_expanded[valid_mol_edge_mask]
        
        mol_node_edge_attr = torch.zeros(num_nodes_mol, 1, device=device, dtype=torch.float32)
        mol_node_edge_attr = mol_node_edge_attr.scatter_add_(0, mol_edge_index_filtered[1].unsqueeze(-1), mol_edge_attr_filtered)
        counts_mol_edges = torch.bincount(mol_edge_index_filtered[1], minlength=num_nodes_mol).float().clamp(min=1).unsqueeze(-1)
        mol_node_edge_attr = mol_node_edge_attr / counts_mol_edges
        mol_edge_attr_pooled = global_max_pool(mol_node_edge_attr, batch.mol_batch)
        
        # Flatten sequence features
        prot_x_protein_pooled = batch.prot_x_protein.view(batch.prot_x_protein.size(0), -1)
        chemBERTa_pooled = batch.chemBERTa.squeeze(1) if batch.chemBERTa.dim() == 3 else batch.chemBERTa
        
        # Project features to scalars
        proj_prot_node_evo = nn.Linear(1280, 1).to(device)(prot_node_evo_pooled)
        proj_prot_node_aa = nn.Linear(9, 1).to(device)(prot_node_aa_pooled)
        prot_x_protein_dim = prot_x_protein_pooled.size(1)
        proj_prot_x_protein = nn.Linear(prot_x_protein_dim, 1).to(device)(prot_x_protein_pooled)
        proj_mol_x_feat = nn.Linear(43, 1).to(device)(mol_x_feat_pooled)
        proj_mol_edge_attr = nn.Linear(1, 1).to(device)(mol_edge_attr_pooled)
        proj_clique_x = nn.Linear(1, 1).to(device)(clique_x_pooled)
        proj_chemBERTa = nn.Linear(768, 1).to(device)(chemBERTa_pooled)
        proj_prot_edge_weight = nn.Linear(1, 1).to(device)(prot_edge_weight_pooled)
        proj_clique_edge_weight = nn.Linear(1, 1).to(device)(clique_edge_weight_pooled)
        
        # Concatenate projected features
        intermediate_features = torch.cat([
            proj_prot_node_evo, proj_prot_node_aa, proj_prot_x_protein,
            proj_mol_x_feat, proj_mol_edge_attr, proj_clique_x, proj_chemBERTa,
            proj_prot_edge_weight, proj_clique_edge_weight
        ], dim=-1)
        # Move to CPU before logging or returning
        intermediate_features = intermediate_features.cpu()
        logging.info(f"intermediate_features shape: {intermediate_features.shape}")
        
    return intermediate_features

# SHAP Analysis Functions
def shap_summary(model, loader, device, log_dir):
    logging.info("Starting SHAP summary computation...")
    model.eval()
    background_batch = next(iter(loader))
    background_features = compute_intermediate_features(background_batch, model, device)
    logging.info(f"Background features computed: {background_features.shape}")
    
    def model_wrapper(features):
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            reg_output, _ = model(intermediate_features=features_tensor)
        return reg_output.cpu().numpy()  # Ensure tensor is on CPU before converting to NumPy
    
    explainer = shap.Explainer(model_wrapper, background_features.numpy())  # background_features is already on CPU
    
    shap_values_list = []
    features_list = []
    id_list = []  # To store sample IDs
    feature_names = [
        'prot_node_evo', 'prot_node_aa', 'prot_x_protein',
        'mol_x_feat', 'mol_edge_attr', 'clique_x', 'chemBERTa',
        'prot_edge_weight', 'clique_edge_weight'
    ]
    
    with torch.no_grad():
        for batch in loader:
            batch_features = compute_intermediate_features(batch, model, device)
            shap_values = explainer(batch_features.numpy())  # batch_features is already on CPU
            shap_values_list.append(shap_values.values)
            features_list.append(batch_features.numpy())
            # Extract IDs from batch
            batch_ids = batch.id if isinstance(batch.id, list) else [batch.id] * batch_features.shape[0]
            id_list.extend(batch_ids)
    
    shap_values = np.concatenate(shap_values_list, axis=0)
    features = np.concatenate(features_list, axis=0)
    logging.info(f"SHAP values shape: {shap_values.shape}")
    logging.info(f"All features shape: {features.shape}")
    logging.info(f"Number of IDs collected: {len(id_list)}")
    
    # Ensure the number of IDs matches the number of samples
    if len(id_list) != shap_values.shape[0]:
        logging.warning(f"ID count ({len(id_list)}) does not match sample count ({shap_values.shape[0]}). Padding with 'Unknown'.")
        id_list = id_list + ['Unknown'] * (shap_values.shape[0] - len(id_list))
    
    # 1. Top 10 samples with greatest overall positive SHAP contribution
    total_positive_shap_per_sample = np.sum(np.maximum(shap_values, 0), axis=1)
    top_10_indices_overall = np.argsort(total_positive_shap_per_sample)[::-1][:10]
    top_10_ids_overall = [id_list[idx] for idx in top_10_indices_overall]
    
    # 2. Top 10 samples for prot_node_aa
    prot_node_aa_idx = feature_names.index('prot_node_aa')  # Index 1
    prot_node_aa_shap = shap_values[:, prot_node_aa_idx]
    positive_prot_node_aa_shap = np.maximum(prot_node_aa_shap, 0)
    top_10_indices_prot_node_aa = np.argsort(positive_prot_node_aa_shap)[::-1][:10]
    top_10_ids_prot_node_aa = [id_list[idx] for idx in top_10_indices_prot_node_aa]
    
    # 3. Top 10 samples for prot_x_protein
    prot_x_protein_idx = feature_names.index('prot_x_protein')  # Index 2
    prot_x_protein_shap = shap_values[:, prot_x_protein_idx]
    positive_prot_x_protein_shap = np.maximum(prot_x_protein_shap, 0)
    top_10_indices_prot_x_protein = np.argsort(positive_prot_x_protein_shap)[::-1][:10]
    top_10_ids_prot_x_protein = [id_list[idx] for idx in top_10_indices_prot_x_protein]
    
    # 4. Top 10 samples for Cross S2G (assuming mol_x_feat)
    cross_s2g_idx = feature_names.index('mol_x_feat')  # Index 3
    cross_s2g_shap = shap_values[:, cross_s2g_idx]
    positive_cross_s2g_shap = np.maximum(cross_s2g_shap, 0)
    top_10_indices_cross_s2g = np.argsort(positive_cross_s2g_shap)[::-1][:10]
    top_10_ids_cross_s2g = [id_list[idx] for idx in top_10_indices_cross_s2g]
    
    # 5. Top 10 samples for Early Fusion (assuming prot_node_evo)
    early_fusion_idx = feature_names.index('prot_node_evo')  # Index 0
    early_fusion_shap = shap_values[:, early_fusion_idx]
    positive_early_fusion_shap = np.maximum(early_fusion_shap, 0)
    top_10_indices_early_fusion = np.argsort(positive_early_fusion_shap)[::-1][:10]
    top_10_ids_early_fusion = [id_list[idx] for idx in top_10_indices_early_fusion]
    
    # Combine into a single DataFrame
    max_len = 10
    data = {
        'Top_10_Overall_ID': top_10_ids_overall + [''] * (max_len - len(top_10_ids_overall)),
        'Top_10_Prot_Node_AA_ID': top_10_ids_prot_node_aa + [''] * (max_len - len(top_10_ids_prot_node_aa)),
        'Top_10_Prot_X_Protein_ID': top_10_ids_prot_x_protein + [''] * (max_len - len(top_10_ids_prot_x_protein)),
        'Top_10_Cross_S2G_ID': top_10_ids_cross_s2g + [''] * (max_len - len(top_10_ids_cross_s2g)),
        'Top_10_Early_Fusion_ID': top_10_ids_early_fusion + [''] * (max_len - len(top_10_ids_early_fusion))
    }
    top_10_df = pd.DataFrame(data)
    csv_path = os.path.join(log_dir, 'top_10_shap_ids.csv')
    top_10_df.to_csv(csv_path, index=False)
    logging.info(f"Top 10 IDs for overall, prot_node_aa, prot_x_protein, Cross S2G, and Early Fusion SHAP contributions saved to {csv_path}")
    
    # Call existing visualization functions
    shap_summary_bar(shap_values, features, feature_names, os.path.join(log_dir, 'shap_summary_test.png'))
    shap_feature_impact(shap_values, feature_names, os.path.join(log_dir, 'shap_feature_impact_test.png'))
    shap_density_scatter(shap_values, features, feature_names, os.path.join(log_dir, 'shap_density_scatter_test.png'))
    shap_hierarchical_clustering(shap_values, features, feature_names, os.path.join(log_dir, 'shap_hierarchical_clustering_test.png'))
    shap_dependence_plots(shap_values, features, feature_names, os.path.join(log_dir, 'shap_test'))
    shap_heatmap(shap_values, os.path.join(log_dir, 'shap_heatmap_test.png'))
    shap_sample_clustering_heatmap(shap_values, feature_names, os.path.join(log_dir, 'shap_sample_clustering_heatmap_test.png'))

def shap_waterfall(model, loader, device, log_dir, sample_idx=0):
    logging.info("Starting SHAP waterfall computation...")
    model.eval()
    background_batch = next(iter(loader))
    background_features = compute_intermediate_features(background_batch, model, device)
    
    def model_wrapper(features):
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            reg_output, _ = model(intermediate_features=features_tensor)
        return reg_output.cpu().numpy()  # Ensure tensor is on CPU before converting to NumPy
    
    explainer = shap.Explainer(model_wrapper, background_features.numpy())
    batch = next(iter(loader))
    batch_features = compute_intermediate_features(batch, model, device)
    
    # Compute SHAP values
    shap_values = explainer(batch_features.numpy())
    
    # Manually compute expected value as the mean prediction over background
    background_predictions = model_wrapper(background_features.numpy())
    expected_value = np.mean(background_predictions)
    
    # Feature names for plotting
    feature_names = [
        'prot_node_evo', 'prot_node_aa', 'prot_x_protein',
        'mol_x_feat', 'mol_edge_attr', 'clique_x', 'chemBERTa',
        'prot_edge_weight', 'clique_edge_weight'
    ]
    
    # Generate waterfall plot
    shap_waterfall_plot(shap_values.values, batch_features.numpy(), feature_names, expected_value, sample_idx, 
                        os.path.join(log_dir, 'shap_waterfall_test.png'))

if __name__ == "__main__":
    try:
        # Load test dataset
        test_dataset = ProteinLigandDataset(
            pkl_path="test.pkl",
            training=False,
            is_classification=True
        )
        batch_size = 4
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        logging.info("Test dataset loaded successfully.")

        # Initialize model
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
        logging.info("Model initialized with specified hyperparameters.")

        # Load pre-trained model
        model_path = "/best_model_multi_task1.pth"
        model.load_state_dict(torch.load(model_path))
        logging.info(f"Pre-trained model loaded from {model_path}")

        # Evaluate model on test set
        logging.info("Evaluating model on test set...")
        metrics = evaluate_model(model, test_loader, device, is_classification=True)
        logging.info("Test evaluation completed.")

        # Log test metrics including AUPRC
        logging.info("Test Classification Metrics:")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}, "
                     f"ROC-AUC: {metrics['roc_auc']:.4f}, "
                     f"F1: {metrics['f1']:.4f}, "
                     f"AUPRC: {metrics['ap']:.4f}")

        # Plot classification visualizations
        logging.info("Generating classification visualizations...")
        dataset_name = "Test"
        plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], dataset_name, log_dir)
        plot_precision_recall_curve(metrics['precision'], metrics['recall'], metrics['ap'], dataset_name, log_dir)
        plot_confusion_matrix(metrics, dataset_name, log_dir)
        plot_classification_metrics(metrics, dataset_name, log_dir)  # Added for Accuracy and F1 visualization
        logging.info("Classification visualizations completed.")

        # Perform SHAP analysis
        logging.info("Starting SHAP analysis...")
        shap_summary(model, test_loader, device, log_dir)
        shap_waterfall(model, test_loader, device, log_dir, sample_idx=0)
        logging.info("SHAP analysis completed.")

    except Exception as e:
        logging.error(f"Testing process failed: {e}")