"""
Pipeline end-to-end: amminoacido → catena di K siti sul reticolo 2D
====================================================================

Uso:
    from amino_lattice import AminoLatticePipeline

    pipeline = AminoLatticePipeline()
    result = pipeline.run(smiles="CC(N)C(=O)O", name="ALA")
    print(result.chain)   # [(i1,j1,'HBondDonor'), ...]
    result.visualize()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Tuple, Optional
import warnings

import numpy as np
from rdkit import Chem

from .feature_extraction import extract_features, mol_from_smiles, AtomFeature
from .site_selection import choose_k, select_representative_sites
from .lattice_fitting import fit_to_lattice_2d, projection_stress
from .snapping import snap_to_lattice
from .labeling import label_sites, encode_chain, LabeledSite


@dataclass
class MappingResult:
    """Risultato della pipeline per un singolo amminoacido."""
    name: str
    smiles: str
    k: int
    features: List[AtomFeature]
    sites: List[AtomFeature]           # K siti aggregati (3D)
    coords_2d: np.ndarray              # (K, 2) float
    lattice_nodes: List[Tuple[int, int]]  # (K,) interi
    labeled_sites: List[LabeledSite]
    stress: float                      # qualità della proiezione 2D

    @property
    def chain(self) -> List[Tuple[int, int, str]]:
        """Formato standard: [(i, j, tipo), ...]"""
        return [(ls.i, ls.j, ls.feature_type) for ls in self.labeled_sites]

    def to_matrix(self) -> np.ndarray:
        """Matrice (K, 2+label_dim) pronta per ML."""
        return encode_chain(self.labeled_sites)

    def summary(self) -> str:
        lines = [
            f"Amminoacido : {self.name}",
            f"SMILES      : {self.smiles}",
            f"Feature raw : {len(self.features)}",
            f"K siti      : {self.k}",
            f"Stress 2D   : {self.stress:.3f}",
            f"Catena      : {self.chain}",
        ]
        return "\n".join(lines)

    def visualize(self, ax=None, show: bool = True):
        """Plotta i K siti sul reticolo 2D."""
        import matplotlib.pyplot as plt
        from .feature_extraction import FEATURE_TYPES

        COLOR_MAP = {
            "HBondDonor":    "#2196F3",
            "HBondAcceptor": "#4CAF50",
            "Hydrophobe":    "#FF9800",
            "Aromatic":      "#9C27B0",
            "PosIonizable":  "#F44336",
            "NegIonizable":  "#00BCD4",
        }

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(6, 6))

        # Disegna griglia
        all_i = [n[0] for n in self.lattice_nodes]
        all_j = [n[1] for n in self.lattice_nodes]
        margin = 2
        for gi in range(min(all_i) - margin, max(all_i) + margin + 1):
            for gj in range(min(all_j) - margin, max(all_j) + margin + 1):
                ax.plot(gj, gi, ".", color="#e0e0e0", markersize=4, zorder=0)

        # Disegna connessioni (catena)
        for k in range(len(self.lattice_nodes) - 1):
            i1, j1 = self.lattice_nodes[k]
            i2, j2 = self.lattice_nodes[k + 1]
            ax.plot([j1, j2], [i1, i2], "-", color="#bbb", lw=1.5, zorder=1)

        # Disegna siti
        for ls in self.labeled_sites:
            color = COLOR_MAP.get(ls.feature_type, "#607D8B")
            ax.scatter(ls.j, ls.i, s=200, color=color, zorder=3,
                       edgecolors="white", linewidths=1.5)
            ax.annotate(
                ls.feature_type[:3],
                (ls.j, ls.i),
                fontsize=7, ha="center", va="center", color="white",
                fontweight="bold", zorder=4,
            )

        ax.set_title(f"{self.name}  (K={self.k}, stress={self.stress:.2f})")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_aspect("equal")

        if created_fig and show:
            plt.tight_layout()
            plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AminoLatticePipeline:
    """
    Orchestratore della pipeline amminoacido → reticolo 2D.

    Parameters
    ----------
    k_strategy : str
        Strategia per la scelta di K ("active_features", "heavy_atoms",
        "fixed", "groups").
    projection_method : str
        "pca" o "mds".
    snap_strategy : str
        "hungarian" (default, no collisioni) o "round".
    label_mode : str
        "one_hot", "index", o "embedding".
    max_k : int
        Cap sul numero di siti.
    lattice_spacing : float
        Angstrom per passo reticolo.
    embed_3d : bool
        Se True genera conformero 3D con ETKDG.
    """

    def __init__(
        self,
        k_strategy: str = "active_features",
        projection_method: str = "pca",
        snap_strategy: str = "hungarian",
        label_mode: str = "one_hot",
        max_k: int = 8,
        min_k: int = 1,
        fixed_k: int = 5,
        lattice_spacing: float = 1.5,
        embed_3d: bool = True,
    ):
        self.k_strategy = k_strategy
        self.projection_method = projection_method
        self.snap_strategy = snap_strategy
        self.label_mode = label_mode
        self.max_k = max_k
        self.min_k = min_k
        self.fixed_k = fixed_k
        self.lattice_spacing = lattice_spacing
        self.embed_3d = embed_3d

    def run(self, smiles: str, name: str = "UNK") -> MappingResult:
        """
        Esegue la pipeline completa su un singolo amminoacido.

        Parameters
        ----------
        smiles : str
            SMILES dell'amminoacido.
        name : str
            Nome per il report (es. "ALA", "TRP").

        Returns
        -------
        MappingResult
        """
        # Step 1: parsing
        mol = mol_from_smiles(smiles)

        # Step 2: estrazione feature
        features = extract_features(mol, embed_3d=self.embed_3d)
        if not features:
            warnings.warn(f"[{name}] Nessuna feature estratta, uso backbone carbonilico")
            features = _backbone_fallback(mol)

        # Step 3: scelta K
        k = choose_k(
            features, mol=mol,
            strategy=self.k_strategy,
            max_k=self.max_k, min_k=self.min_k,
            fixed_k=self.fixed_k,
        )

        # Selezione K siti rappresentativi
        sites = select_representative_sites(features, k)
        k = len(sites)  # può essere < k se feature < k

        # Step 4: fitting geometrico
        coords_2d = fit_to_lattice_2d(
            sites,
            method=self.projection_method,
            lattice_spacing=self.lattice_spacing,
        )

        # Stress (qualità proiezione)
        stress = projection_stress(sites, coords_2d)

        # Step 5: snapping
        lattice_nodes = snap_to_lattice(coords_2d, strategy=self.snap_strategy)

        # Step 6: labeling
        labeled_sites = label_sites(sites, lattice_nodes, mode=self.label_mode)

        return MappingResult(
            name=name,
            smiles=smiles,
            k=k,
            features=features,
            sites=sites,
            coords_2d=coords_2d,
            lattice_nodes=lattice_nodes,
            labeled_sites=labeled_sites,
            stress=stress,
        )

    def run_batch(self, records: list[dict]) -> list[MappingResult]:
        """
        Processa un batch di amminoacidi.

        Parameters
        ----------
        records : list of dict con chiavi "smiles" e "name"

        Returns
        -------
        list of MappingResult
        """
        from tqdm import tqdm
        results = []
        for rec in tqdm(records, desc="Mapping"):
            try:
                r = self.run(smiles=rec["smiles"], name=rec.get("name", "UNK"))
                results.append(r)
            except Exception as e:
                warnings.warn(f"[{rec.get('name','?')}] Errore: {e}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Fallback
# ─────────────────────────────────────────────────────────────────────────────

def _backbone_fallback(mol) -> List[AtomFeature]:
    """Se non si trovano feature farmacofore, usa gli atomi del backbone."""
    from .feature_extraction import AtomFeature
    from rdkit import Chem
    mol_no_h = Chem.RemoveHs(mol)
    from rdkit.Chem import rdDepictor
    rdDepictor.Compute2DCoords(mol_no_h)
    conf = mol_no_h.GetConformer()
    features = []
    for atom in mol_no_h.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        features.append(AtomFeature(
            feature_type="HBondDonor",
            coords=np.array([pos.x, pos.y, pos.z]),
            atom_indices=[atom.GetIdx()],
        ))
    return features
