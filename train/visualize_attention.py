# visualize_attention.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
import shap
from scipy.cluster import hierarchy
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.colors as mcolors

def visualize_all_attentions(model, batch, save_path):
    """
    Visualize a heatmap where each cell represents the interaction between attention mechanisms,
    with mechanisms as axis labels. Off-diagonal entries show cosine similarity between mechanisms.
    Higher weights are red, lower weights are blue, with numerical annotations.

    Args:
        model: The model containing attention weights
        batch: The input batch
        save_path: Path to save the plot
    """
    logging.info("Starting visualize_all_attentions...")
    attn_data = []
    attn_matrices = []  # Store raw attention matrices for similarity computation

    # Helper function to flatten attention weights for similarity computation
    def flatten_weights(weights):
        weights_np = weights.detach().cpu().numpy()
        if weights_np.ndim > 2:
            weights_np = weights_np.mean(axis=tuple(range(1, weights_np.ndim - 1)))  # Average over heads, etc.
        if weights_np.ndim == 2:
            return weights_np.flatten()
        return weights_np

    # Protein Graph Encoder (GAT1 and GAT2)
    if hasattr(model.prot_graph_enc, 'gat1') and model.prot_graph_enc.gat1_attn_weights is not None:
        gat1_weights = model.prot_graph_enc.gat1_attn_weights  # [num_edges, num_heads]
        if gat1_weights.ndim > 1:
            gat1_weights = gat1_weights.mean(dim=1)  # Average over heads
        mean_weight = gat1_weights.mean().item()
        attn_data.append(("Prot GAT1", mean_weight))
        attn_matrices.append(flatten_weights(gat1_weights))

    if hasattr(model.prot_graph_enc, 'gat2') and model.prot_graph_enc.gat2_attn_weights is not None:
        gat2_weights = model.prot_graph_enc.gat2_attn_weights  # [num_edges, num_heads]
        if gat2_weights.ndim > 1:
            gat2_weights = gat2_weights.mean(dim=1)  # Average over heads
        mean_weight = gat2_weights.mean().item()
        attn_data.append(("Prot GAT2", mean_weight))
        attn_matrices.append(flatten_weights(gat2_weights))

    # Ligand Graph Encoder (GAT Mol and GAT Clique)
    if hasattr(model.lig_graph_enc, 'gat_mol') and model.lig_graph_enc.gat_mol_attn_weights is not None:
        gat_mol_weights = model.lig_graph_enc.gat_mol_attn_weights  # [num_edges, num_heads]
        if gat_mol_weights.ndim > 1:
            gat_mol_weights = gat_mol_weights.mean(dim=1)  # Average over heads
        mean_weight = gat_mol_weights.mean().item()
        attn_data.append(("Lig GAT Mol", mean_weight))
        attn_matrices.append(flatten_weights(gat_mol_weights))

    if hasattr(model.lig_graph_enc, 'gat_clique') and model.lig_graph_enc.gat_clique_attn_weights is not None:
        gat_clique_weights = model.lig_graph_enc.gat_clique_attn_weights  # [num_edges, num_heads]
        if gat_clique_weights.ndim > 1:
            gat_clique_weights = gat_clique_weights.mean(dim=1)  # Average over heads
        mean_weight = gat_clique_weights.mean().item()
        attn_data.append(("Lig GAT Clique", mean_weight))
        attn_matrices.append(flatten_weights(gat_clique_weights))

    # Ligand Graph Encoder (ChemBERTa Transformer)
    if hasattr(model.lig_graph_enc, 'chemberta_attn_weights') and model.lig_graph_enc.chemberta_attn_weights:
        for layer_idx, chemberta_weights in enumerate(model.lig_graph_enc.chemberta_attn_weights):
            chemberta_weights_np = chemberta_weights.detach().cpu().numpy()  # [batch_size, num_heads, seq_len, seq_len]
            if chemberta_weights_np.ndim > 3:
                chemberta_weights_np = chemberta_weights_np.mean(axis=1)  # Average over heads
            chemberta_weights_np = chemberta_weights_np[0]  # First batch element
            mean_weight = chemberta_weights_np.mean()
            attn_data.append((f"ChemBERTa L{layer_idx+1}", mean_weight))
            attn_matrices.append(chemberta_weights_np.flatten())

    # Ligand Graph Encoder (Fusion Attention)
    if hasattr(model.lig_graph_enc, 'fusion_attn_weights') and model.lig_graph_enc.fusion_attn_weights is not None:
        fusion_weights = model.lig_graph_enc.fusion_attn_weights  # [num_queries, num_keys, num_keys]
        fusion_weights_np = fusion_weights.detach().cpu().numpy()
        if fusion_weights_np.ndim > 2:
            fusion_weights_np = fusion_weights_np.mean(axis=0)  # Average over heads
        mean_weight = fusion_weights_np.mean()
        attn_data.append(("Lig Fusion", mean_weight))
        attn_matrices.append(fusion_weights_np.flatten())

    # Protein Sequence Encoder (self-attention)
    if hasattr(model.prot_seq_enc, 'attn_weights') and model.prot_seq_enc.attn_weights is not None:
        prot_seq_weights = model.prot_seq_enc.attn_weights  # [seq_len, batch_size, seq_len]
        prot_seq_weights_np = prot_seq_weights.detach().cpu().numpy()
        if prot_seq_weights_np.ndim > 2:
            prot_seq_weights_np = prot_seq_weights_np.mean(axis=1)  # Average over heads
        mean_weight = prot_seq_weights_np.mean()
        attn_data.append(("Prot Seq", mean_weight))
        attn_matrices.append(prot_seq_weights_np.flatten())

    # CrossTransformerFusion (g2s and s2g attention)
    if hasattr(model.cross_modal_fusion, 'g2s_attn_weights'):
        for i, g2s_weights in enumerate(model.cross_modal_fusion.g2s_attn_weights):
            if g2s_weights is not None:
                g2s_weights_np = g2s_weights.detach().cpu().numpy()  # [num_queries, batch_size, num_keys]
                if g2s_weights_np.ndim > 2:
                    g2s_weights_np = g2s_weights_np.mean(axis=1)  # Average over heads
                mean_weight = g2s_weights_np.mean()
                attn_data.append((f"Cross G2S L{i+1}", mean_weight))
                attn_matrices.append(g2s_weights_np.flatten())

    if hasattr(model.cross_modal_fusion, 's2g_attn_weights'):
        for i, s2g_weights in enumerate(model.cross_modal_fusion.s2g_attn_weights):
            if s2g_weights is not None:
                s2g_weights_np = s2g_weights.detach().cpu().numpy()  # [num_queries, batch_size, num_keys]
                if s2g_weights_np.ndim > 2:
                    s2g_weights_np = s2g_weights_np.mean(axis=1)  # Average over heads
                mean_weight = s2g_weights_np.mean()
                attn_data.append((f"Cross S2G L{i+1}", mean_weight))
                attn_matrices.append(g2s_weights_np.flatten())

    # Early Fusion Attention
    if hasattr(model, 'early_fusion_attn_weights') and model.early_fusion_attn_weights is not None:
        early_fusion_weights = model.early_fusion_attn_weights  # [num_queries, batch_size, num_keys]
        early_fusion_weights_np = early_fusion_weights.detach().cpu().numpy()
        if early_fusion_weights_np.ndim > 2:
            early_fusion_weights_np = early_fusion_weights_np.mean(axis=1)  # Average over heads
        mean_weight = early_fusion_weights_np.mean()
        attn_data.append(("Early Fusion", mean_weight))
        attn_matrices.append(early_fusion_weights_np.flatten())

    # Check if any attention data was collected
    if not attn_data:
        logging.warning("No valid attention weights available to visualize.")
        return

    # Create a matrix of mechanism-level interactions
    num_mechanisms = len(attn_data)
    mechanism_matrix = np.zeros((num_mechanisms, num_mechanisms))
    mechanism_names = [name for name, _ in attn_data]

    # Compute cosine similarity between mechanisms for off-diagonal entries
    for i in range(num_mechanisms):
        for j in range(num_mechanisms):
            if i == j:
                # Diagonal: use the mean attention weight
                mechanism_matrix[i, j] = attn_data[i][1]
            else:
                # Off-diagonal: compute cosine similarity between attention weights
                vec_i = attn_matrices[i].reshape(1, -1)
                vec_j = attn_matrices[j].reshape(1, -1)
                # Ensure vectors are the same length by padding/truncating
                max_len = max(vec_i.shape[1], vec_j.shape[1])
                vec_i_padded = np.pad(vec_i, ((0, 0), (0, max_len - vec_i.shape[1])), mode='constant')
                vec_j_padded = np.pad(vec_j, ((0, 0), (0, max_len - vec_j.shape[1])), mode='constant')
                similarity = cosine_similarity(vec_i_padded, vec_j_padded)[0, 0]
                mechanism_matrix[i, j] = similarity

    # Normalize the matrix to [-1, 1] for the RdBu colormap
    min_val = min(mechanism_matrix.min(), -1.0)
    max_val = max(mechanism_matrix.max(), 1.0)
    if max_val != min_val:
        mechanism_matrix = 2 * (mechanism_matrix - min_val) / (max_val - min_val) - 1  # Scale to [-1, 1]

    # Create the heatmap
    logging.info("Generating mechanism-level attention heatmap...")
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        mechanism_matrix,
        cmap="RdBu_r",
        annot=mechanism_matrix,
        fmt=".2f",
        cbar=True,
        xticklabels=mechanism_names,
        yticklabels=mechanism_names,
        square=True,
        annot_kws={"size": 8}
    )

    # Customize annotation colors to match the RdBu colormap
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    for text in ax.texts:
        value = float(text.get_text())
        color = cmap(norm(value))
        if value > 0.5:
            color = 'white'
        elif value < -0.5:
            color = 'black'
        text.set_color(color)

    plt.title("Attention Mechanism Heatmap")
    plt.xlabel("Target Mechanism")
    plt.ylabel("Source Mechanism")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Mechanism-level attention visualization saved to {save_path}")

# SHAP Visualization Functions
def shap_summary_bar(shap_values, features, feature_names, save_path):
    """Generate a SHAP summary plot (bar type) for all features."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features=features, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"SHAP summary bar plot saved to {save_path}")

def shap_waterfall_plot(shap_values, features, feature_names, expected_value, sample_idx, save_path):
    """Generate a SHAP waterfall plot for a specific sample."""
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        data=features[sample_idx],
        feature_names=feature_names
    )
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(explanation, show=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logging.info(f"SHAP waterfall plot saved to {save_path}")

def shap_density_scatter(shap_values, features, feature_names, save_path):
    """
    Generate a SHAP summary plot (beeswarm style) for the given SHAP values and features.
    
    Args:
        shap_values (np.ndarray): SHAP values, shape [num_samples, num_features].
        features (np.ndarray): Feature values, shape [num_samples, num_features].
        feature_names (list): Names of the features.
        save_path (str): Path to save the plot.
    """
    shap_exp = shap.Explanation(
        values=shap_values,
        data=features,
        feature_names=feature_names
    )
    plt.figure(figsize=(10, len(feature_names) * 0.5))
    shap.summary_plot(
        shap_exp,
        features=features,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=len(feature_names)
    )
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("Feature value", rotation=270, labelpad=15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"SHAP density scatter plot (beeswarm style) saved to {save_path}")

def shap_hierarchical_clustering(shap_values, features, feature_names, save_path):
    """Generate a hierarchical clustering dendrogram of features based on SHAP values."""
    correlation_matrix = np.corrcoef(shap_values.T)
    linkage = hierarchy.linkage(correlation_matrix, method='average')
    plt.figure(figsize=(12, 8))
    hierarchy.dendrogram(linkage, labels=feature_names, leaf_rotation=45, leaf_font_size=10)
    plt.title("Hierarchical Clustering of Features by SHAP Correlation")
    plt.xlabel("Features")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"SHAP hierarchical clustering plot saved to {save_path}")

def shap_dependence_plots(shap_values, features, feature_names, save_path_prefix):
    """
    Generate dependence plots for specified feature pairs to reflect relationships:
    - Protein-Protein: 'prot_node_evo' vs 'prot_x_protein'
    - Ligand-Ligand: 'mol_x_feat' vs 'chemBERTa'
    - Protein-Ligand: 'prot_node_evo' vs 'mol_x_feat'
    Also include original features 'prot_x_protein' and 'clique_x'.
    """
    # Define the feature pairs to be generated
    feature_pairs = [
        # Protein-Protein
        ('prot_node_evo', 'prot_x_protein'),
        # Ligand-Ligand
        ('mol_x_feat', 'chemBERTa'),
        # Protein-Ligand
        ('prot_node_evo', 'mol_x_feat'),
        # Original features
        ('prot_x_protein', None),
        ('clique_x', None)
    ]
    
    for main_feat, interact_feat in feature_pairs:
        main_idx = feature_names.index(main_feat)
        if interact_feat:
            interact_idx = feature_names.index(interact_feat)
            plot_name = f"{main_feat}_vs_{interact_feat}"
        else:
            interact_idx = None
            plot_name = main_feat
            
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            ind=main_idx,
            shap_values=shap_values,
            features=features,
            feature_names=feature_names,
            interaction_index=interact_idx,
            show=False
        )
        save_path = f"{save_path_prefix}_{plot_name}_dependence.png"
        plt.savefig(save_path)
        plt.close()
        logging.info(f"SHAP dependence plot for {plot_name} saved to {save_path}")

def shap_force_plot(shap_values, features, feature_names, expected_value, save_path):
    """
    Generate a SHAP force plot for the first sample in the dataset to avoid matplotlib=True limitation.

    Args:
        shap_values (np.ndarray): SHAP values, shape [num_samples, num_features].
        features (np.ndarray): Feature values, shape [num_samples, num_features].
        feature_names (list): Names of the features.
        expected_value (float): Expected value (base value) of the model output.
        save_path (str): Path to save the plot.
    """
    if shap_values.shape[0] == 0:
        logging.warning("No samples available for SHAP force plot.")
        return

    # Generate force plot for first sample only
    shap_values_single = shap_values[0]  #
    features_single = features[0]  

    plt.figure(figsize=(20, 3))
    shap.force_plot(
        base_value=expected_value,
        shap_values=shap_values_single,
        features=features_single,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logging.info(f"SHAP force plot for the first sample saved to {save_path}")

def shap_decision_plot(shap_values, features, feature_names, expected_value, save_path, num_samples=50):
    """Generate a SHAP decision plot for multiple samples."""
    
    num_samples = min(num_samples, shap_values.shape[0])
    shap_values_subset = shap_values[:num_samples]
    features_subset = features[:num_samples]
    
    plt.figure(figsize=(12, 8))
    shap.decision_plot(
        base_value=expected_value,
        shap_values=shap_values_subset,
        features=features_subset,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logging.info(f"SHAP decision plot for {num_samples} samples saved to {save_path}")

def shap_heatmap_plot(shap_values, features, feature_names, save_path):
    """
    Generate a SHAP heatmap plot for the entire dataset.
    
    Args:
        shap_values (np.ndarray): SHAP values, shape [num_samples, num_features].
        features (np.ndarray): Feature values, shape [num_samples, num_features].
        feature_names (list): Names of the features.
        save_path (str): Path to save the plot.
    """
    # Creating a SHAP Explanation Object
    shap_exp = shap.Explanation(
        values=shap_values,
        data=features,
        feature_names=feature_names
    )
    
    plt.figure(figsize=(12, 8))
    shap.plots.heatmap(shap_exp, show=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logging.info(f"SHAP heatmap plot saved to {save_path}")