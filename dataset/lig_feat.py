import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors, ChemicalFeatures
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import BondType
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, to_undirected
from itertools import chain
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Initialize chemical feature factory
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

# Helper functions
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    encoding = (one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2, 'other']) +
                [atom.GetIsAromatic()])
    try:
        encoding += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
    return np.array(encoding)

# Validate SMILES string
def validate_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Standardize SMILES sequence
def standardize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    normalizer = rdMolStandardize.Normalizer()
    standardized_mol = normalizer.normalize(mol)
    return Chem.MolToSmiles(standardized_mol)

# MoleculeGraphDataset class
class MoleculeGraphDataset:
    def __init__(self, atom_classes=None, halogen_detail=False, save_path=None):
        self.ATOM_CODES = {}
        if atom_classes is None:
            metals = ([3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) +
                      list(range(55, 84)) + list(range(87, 104)))
            self.FEATURE_NAMES = []
            if halogen_detail:
                atom_classes = [
                    (5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'), (16, 'S'), (34, 'Se'),
                    (9, 'F'), (17, 'Cl'), (35, 'Br'), (53, 'I'), (metals, 'metal')
                ]
            else:
                atom_classes = [
                    (5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'), (16, 'S'), (34, 'Se'),
                    ([9, 17, 35, 53], 'halogen'), (metals, 'metal')
                ]
        self.NUM_ATOM_CLASSES = len(atom_classes)
        for code, (atom, name) in enumerate(atom_classes):
            if isinstance(atom, list):
                for a in atom:
                    self.ATOM_CODES[a] = code
            else:
                self.ATOM_CODES[atom] = code
            self.FEATURE_NAMES.append(name)
        self.feat_types = ['Donor', 'Acceptor', 'Hydrophobe', 'LumpedHydrophobe']
        self.edge_dict = {
            BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3,
            BondType.AROMATIC: 4, BondType.UNSPECIFIED: 1
        }
        self.save_path = save_path

    def mol_full_feature(self, mol):
        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            feature = atom_features(atom)
            atom_feats.append(feature)
        return np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1])

    def bond_feature(self, mol):
        atom_num = len(mol.GetAtoms())
        adj = np.zeros((atom_num, atom_num))
        for b in mol.GetBonds():
            v1 = b.GetBeginAtomIdx()
            v2 = b.GetEndAtomIdx()
            b_type = self.edge_dict.get(b.GetBondType(), 1)
            adj[v1, v2] = b_type
            adj[v2, v1] = b_type
        return adj

    def junction_tree(self, mol):
        tree_edge_index, atom2clique_index, num_cliques, x_clique = tree_decomposition(mol, return_vocab=True)
        if atom2clique_index.nelement() == 0:
            num_cliques = len(mol.GetAtoms())
            x_clique = torch.tensor([3] * num_cliques)
            atom2clique_index = torch.stack([torch.arange(num_cliques), torch.arange(num_cliques)])
        return dict(tree_edge_index=tree_edge_index, atom2clique_index=atom2clique_index,
                    num_cliques=num_cliques, x_clique=x_clique)

# Tree Decomposition function
def tree_decomposition(mol, return_vocab=False):
    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    xs = [0] * len(cliques)
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            xs.append(1)
    atom2clique = [[] for _ in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)
    for c1 in range(len(cliques)):
        for atom in cliques[c1]:
            for c2 in atom2clique[atom]:
                if c1 >= c2 or len(cliques[c1]) <= 2 or len(cliques[c2]) <= 2:
                    continue
                if len(set(cliques[c1]) & set(cliques[c2])) > 2:
                    cliques[c1] = set(cliques[c1]) | set(cliques[c2])
                    xs[c1] = 2
                    cliques[c2] = []
                    xs[c2] = -1
    cliques = [c for c in cliques if len(c) > 0]
    xs = [x for x in xs if x >= 0]
    atom2clique = [[] for _ in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)
    edges = {}
    for atom in range(mol.GetNumAtoms()):
        cs = atom2clique[atom]
        if len(cs) <= 1:
            continue
        bonds = [c for c in cs if len(cliques[c]) == 2]
        rings = [c for c in cs if len(cliques[c]) > 4]
        if len(bonds) > 2 or (len(bonds) == 2 and len(cs) > 2):
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 99
        else:
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    c1, c2 = cs[i], cs[j]
                    count = len(set(cliques[c1]) & set(cliques[c2]))
                    edges[(c1, c2)] = min(count, edges.get((c1, c2), 99))
    atom2clique = [[] for _ in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)
    if len(edges) > 0:
        edge_index_T, weight = zip(*edges.items())
        edge_index = torch.tensor(edge_index_T).t()
        inv_weight = 100 - torch.tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(cliques))
        junc_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(junc_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(cliques))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    rows = [[i] * len(atom2clique[i]) for i in range(mol.GetNumAtoms())]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(atom2clique)))
    atom2clique = torch.stack([row, col], dim=0).to(torch.long)
    if return_vocab:
        vocab = torch.tensor(xs, dtype=torch.long)
        return edge_index, atom2clique, len(cliques), vocab
    else:
        return edge_index, atom2clique, len(cliques)

# Data Preprocessing class
class JTVAEDataPreprocessor:
    def __init__(self, model_path: str, max_seq_length: int = 200):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        self.max_seq_length = max_seq_length
        self.scaler = StandardScaler()
        self.linear_layer = nn.Linear(self.model.config.hidden_size, 256)  # Linear layer for dimensionality reduction

    def process(self, smiles: str, halogen_detail: bool = False):
        try:
            logger.info(f"Start processing SMILES: {smiles[:50]}...")
            # Standardize SMILES sequence
            standardized_smiles = standardize_smiles(smiles)
            if standardized_smiles is None:
                raise ValueError(f"Failed to standardize SMILES: {smiles}")

            mol = Chem.MolFromSmiles(standardized_smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES after standardization: {standardized_smiles}")

            mgd = MoleculeGraphDataset(halogen_detail=halogen_detail)
            atom_features = mgd.mol_full_feature(mol)
            bond_features = mgd.bond_feature(mol)
            tree = mgd.junction_tree(mol)

            # Extract ChemBERTa features
            inputs = self.tokenizer(standardized_smiles, return_tensors='pt', max_length=self.max_seq_length,
                                    padding='max_length', truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                chemberta_features = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # Mean pooling

            chemBERTa = torch.tensor(chemberta_features, dtype=torch.float32)

            # Extract Morgan fingerprint
            morgan_fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            morgan_fingerprint = torch.tensor([int(bit) for bit in morgan_fingerprint.ToBitString()],
                                             dtype=torch.float32).unsqueeze(0)

            # Convert bond features to PyTorch tensor
            bond_features_tensor = torch.tensor(bond_features, dtype=torch.float32)
            edge_index = torch.nonzero(bond_features_tensor, as_tuple=False).t()
            edge_weight = bond_features_tensor[bond_features_tensor > 0]

            # Extract Junction Tree information
            tree_edge_index = tree['tree_edge_index']
            tree_edge_weight = torch.ones(tree_edge_index.shape[1])  # Default weight of 1
            num_cliques = tree['num_cliques']
            atom2clique_index = tree['atom2clique_index']
            x_clique = tree['x_clique']

            # Return processed data with standardized SMILES as "seq"
            result = {
                "mol_x": torch.tensor([atom.GetIdx() for atom in mol.GetAtoms()], dtype=torch.long).view(-1, 1),
                "seq": standardized_smiles,  # Changed to store standardized SMILES string
                "seq_len": len(standardized_smiles),  # Length of the standardized SMILES string
                "atom_feature": torch.tensor(atom_features, dtype=torch.float32),
                "chemBERTa": chemBERTa,
                "morgan_fingerprint": morgan_fingerprint,
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "tree_edge_index": tree_edge_index,
                "tree_edge_weight": tree_edge_weight,
                "num_cliques": num_cliques,
                "atom2clique_index": atom2clique_index,
                "atom_idx": torch.tensor([atom.GetIdx() for atom in mol.GetAtoms()], dtype=torch.long),
                "bond_feature": bond_features_tensor,
                "x_clique": x_clique
            }
            logger.info(f"SMILES processing completed: {smiles[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Failed to process SMILES '{smiles}': {e}")
            return None

# Process SMILES file
def process_smiles_file(file_path: str, model_path: str, max_seq_length: int = 200):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    preprocessor = JTVAEDataPreprocessor(model_path, max_seq_length=max_seq_length)
    
    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(smiles_list)} SMILES strings from {file_path}")
    
    results = []
    for idx, smiles in enumerate(smiles_list):
        print(f"\nProcessing SMILES {idx + 1}/{len(smiles_list)}: {smiles}")
        if not validate_smiles(smiles):
            print(f"Invalid SMILES string: {smiles}. Skipping...")
            continue
        try:
            graph_data = preprocessor.process(smiles, halogen_detail=False)
            if graph_data is not None:
                results.append(graph_data)
                print(f"Processed successfully: {smiles}")
            else:
                print(f"Failed to process {smiles}")
        except Exception as e:
            print(f"Failed to process {smiles}: {e}")
    
    return results

if __name__ == "__main__":
    smiles_file = "/smiles.txt"
    model_path = "/chemberta_model/"
    results = process_smiles_file(smiles_file, model_path)
    print("\nFinal Results:")
    for result in results:
        print(result)