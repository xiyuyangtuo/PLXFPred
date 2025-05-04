import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from itertools import chain
from scipy.sparse.csgraph import minimum_spanning_tree
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, to_undirected

# Initialize the Chemical Feature Factory
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

class MoleculeGraph:
    def __init__(self, atom_classes=None, halogen_detail=False):
        """
        Initialize the MoleculeGraph class for converting SMILES strings to graph representations.

        Args:
            atom_classes (list, optional): Custom atom classes for feature encoding.
            halogen_detail (bool): Whether to treat halogens as separate classes or a single group.
        """
        self.ATOM_CODES = {}
        self.FEATURE_NAMES = []

        # Define atom classes for encoding
        if atom_classes is None:
            metals = ([3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + list(range(55, 84)) + list(range(87, 104)))
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

        # Map atom numbers to class codes
        self.NUM_ATOM_CLASSES = len(atom_classes)
        for code, (atom, name) in enumerate(atom_classes):
            if isinstance(atom, list):
                for a in atom:
                    self.ATOM_CODES[a] = code
            else:
                self.ATOM_CODES[atom] = code
            self.FEATURE_NAMES.append(name)

        # Define bond types
        self.edge_dict = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4,
            Chem.rdchem.BondType.UNSPECIFIED: 1
        }

    def one_of_k_encoding(self, x, allowable_set):
        """Helper function for one-hot encoding."""
        if x not in allowable_set:
            raise ValueError(f"Input {x} not in allowable set {allowable_set}")
        return [x == s for s in allowable_set]

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Helper function for one-hot encoding with an unknown category."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]

    def atom_features(self, atom):
        """Extract features for a single atom."""
        encoding = self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                   self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        encoding += self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        encoding += self.one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, 'other'])
        encoding += [atom.GetIsAromatic()]
        try:
            encoding += self.one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]
        return np.array(encoding)

    def mol_full_feature(self, mol):
        """Extract full atom features for a molecule."""
        atom_ids = []
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            feature = self.atom_features(atom)
            atom_feats.append(feature)
        return np.array([feat for _, feat in sorted(zip(atom_ids, atom_feats))])

    def bond_feature(self, mol):
        """Extract bond features (adjacency matrix) for a molecule."""
        atom_num = len(mol.GetAtoms())
        adj = np.zeros((atom_num, atom_num))
        for bond in mol.GetBonds():
            v1 = bond.GetBeginAtomIdx()
            v2 = bond.GetEndAtomIdx()
            b_type = self.edge_dict[bond.GetBondType()]
            adj[v1, v2] = b_type
            adj[v2, v1] = b_type
        return adj

    def get_atom_types(self, mol):
        """Extract atom types for a molecule."""
        atom_types = []
        for atom in mol.GetAtoms():
            atom_num = atom.GetAtomicNum()
            if atom_num not in self.ATOM_CODES:
                raise ValueError(f"Atom number {atom_num} not in ATOM_CODES")
            atom_type = self.ATOM_CODES[atom_num]
            atom_types.append(atom_type)
        return np.array(atom_types)

    def tree_decomposition(self, mol, return_vocab=False):
        """Perform tree decomposition on a molecule to create a junction tree."""
        # Step 1: Identify cliques (SSSR rings and non-ring bonds)
        cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
        xs = [0] * len(cliques)  # 0 for ring cliques
        for bond in mol.GetBonds():
            if not bond.IsInRing():
                cliques.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                xs.append(1)  # 1 for bond cliques

        # Step 2: Map atoms to cliques
        atom2clique = [[] for _ in range(mol.GetNumAtoms())]
        for c in range(len(cliques)):
            for atom in cliques[c]:
                atom2clique[atom].append(c)

        # Step 3: Merge cliques with large intersections
        for c1 in range(len(cliques)):
            for atom in cliques[c1]:
                for c2 in atom2clique[atom]:
                    if c1 >= c2 or len(cliques[c1]) <= 2 or len(cliques[c2]) <= 2:
                        continue
                    if len(set(cliques[c1]) & set(cliques[c2])) > 2:
                        cliques[c1] = set(cliques[c1]) | set(cliques[c2])
                        xs[c1] = 2  # 2 for merged cliques
                        cliques[c2] = []
                        xs[c2] = -1  # Mark as removed

        # Filter out removed cliques
        cliques = [c for c in cliques if len(c) > 0]
        xs = [x for x in xs if x >= 0]

        # Rebuild atom-to-clique mapping
        atom2clique = [[] for _ in range(mol.GetNumAtoms())]
        for c in range(len(cliques)):
            for atom in cliques[c]:
                atom2clique[atom].append(c)

        # Step 4: Build edges between cliques
        edges = {}
        for atom in range(mol.GetNumAtoms()):
            cs = atom2clique[atom]
            if len(cs) <= 1:
                continue
            bonds = [c for c in cs if len(cliques[c]) == 2]
            rings = [c for c in cs if len(cliques[c]) > 4]
            if len(bonds) > 2 or (len(bonds) == 2 and len(cs) > 2):
                cliques.append([atom])
                xs.append(3)  # 3 for single-atom clique
                c2 = len(cliques) - 1
                for c1 in cs:
                    edges[(c1, c2)] = 1
            elif len(rings) > 2:
                cliques.append([atom])
                xs.append(3)  # 3 for single-atom clique
                c2 = len(cliques) - 1
                for c1 in cs:
                    edges[(c1, c2)] = 99
            else:
                for i in range(len(cs)):
                    for j in range(i + 1, len(cs)):
                        c1, c2 = cs[i], cs[j]
                        count = len(set(cliques[c1]) & set(cliques[c2]))
                        edges[(c1, c2)] = min(count, edges.get((c1, c2), 99))

        # Rebuild atom-to-clique mapping again
        atom2clique = [[] for _ in range(mol.GetNumAtoms())]
        for c in range(len(cliques)):
            for atom in cliques[c]:
                atom2clique[atom].append(c)

        # Step 5: Compute the junction tree using a minimum spanning tree
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

        # Step 6: Create atom-to-clique index tensor
        rows = [[i] * len(atom2clique[i]) for i in range(mol.GetNumAtoms())]
        row = torch.tensor(list(chain.from_iterable(rows)))
        col = torch.tensor(list(chain.from_iterable(atom2clique)))
        atom2clique = torch.stack([row, col], dim=0).to(torch.long)

        if return_vocab:
            vocab = torch.tensor(xs, dtype=torch.long)
            return edge_index, atom2clique, len(cliques), vocab
        else:
            return edge_index, atom2clique, len(cliques)

    def smiles2graph(self, smiles):
        """
        Convert a SMILES string to a graph representation.

        Args:
            smiles (str): The SMILES string representing the molecule.

        Returns:
            dict: A dictionary containing the graph representation with the following keys:
                - 'smiles': The input SMILES string.
                - 'atom_feature': Tensor of atom features.
                - 'atom_types': A '|' separated string of atom symbols.
                - 'atom_idx': Tensor of atom type indices.
                - 'bond_feature': Tensor of bond features (adjacency matrix).
                - 'bond_weights': Tensor of bond weights (for visualization).
                - 'tree_edge_index': Tensor of junction tree edge indices.
                - 'atom2clique_index': Tensor mapping atoms to cliques.
                - 'num_cliques': Number of cliques in the junction tree.
                - 'x_clique': Tensor of clique types.
        """
        # Validate SMILES
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("Invalid SMILES string: must be a non-empty string")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES string: {smiles}")

        # Extract features
        atom_feature = self.mol_full_feature(mol)
        bond_feature = self.bond_feature(mol)
        atom_idx = self.get_atom_types(mol)
        tree = self.tree_decomposition(mol, return_vocab=True)

        # Extract bond weights for visualization
        bond_weights = np.zeros_like(bond_feature)
        for bond in mol.GetBonds():
            v1 = bond.GetBeginAtomIdx()
            v2 = bond.GetEndAtomIdx()
            bond_weights[v1, v2] = bond.GetBondTypeAsDouble()
            bond_weights[v2, v1] = bond.GetBondTypeAsDouble()

        # Construct output dictionary
        out_dict = {
            'smiles': smiles,
            'atom_feature': torch.tensor(atom_feature, dtype=torch.float),
            'atom_types': '|'.join([atom.GetSymbol() for atom in mol.GetAtoms()]),
            'atom_idx': torch.tensor(atom_idx, dtype=torch.long),
            'bond_feature': torch.tensor(bond_feature, dtype=torch.float),
            'bond_weights': torch.tensor(bond_weights, dtype=torch.float),
            'tree_edge_index': tree[0],
            'atom2clique_index': tree[1],
            'num_cliques': tree[2],
            'x_clique': tree[3],
        }
        return out_dict

if __name__ == "__main__":
    # Example usage
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin SMILES
    mol_graph = MoleculeGraph(halogen_detail=False)
    graph_data = mol_graph.smiles2graph(smiles)
    print("Graph data keys:", graph_data.keys())
    print("Atom features shape:", graph_data['atom_feature'].shape)
    print("Bond features shape:", graph_data['bond_feature'].shape)
    print("Number of cliques:", graph_data['num_cliques'])