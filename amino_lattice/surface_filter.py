"""
Surface Filter — filtra le feature farmacofore per accessibilità al solvente (SASA).

Motivazione biologica:
    Nel molecular docking il ligando interagisce solo con i gruppi funzionali
    esposti verso l'interno della tasca (accessibili al solvente). Le feature
    farmacofore sepolte nel core proteico non sono fisicamente raggiungibili
    dal ligando e introducono rumore nella rappresentazione.

    Il criterio usato è il SASA (Solvent Accessible Surface Area), calcolato
    con l'algoritmo di FreeSASA tramite BioPython. Un atomo con SASA > soglia
    è considerato esposto; la feature viene mantenuta solo se almeno uno dei
    suoi atomi costituenti è esposto.

Soglie di riferimento (letteratura):
    - Atomo completamente esposto:  SASA relativa > 25%  (rispetto al max teorico)
    - Soglia assoluta conservativa: SASA > 1.0 Å²  (qualsiasi esposizione)
    - Soglia più restrittiva:       SASA > 5.0 Å²  (esposizione significativa)

    Usiamo 1.0 Å² come default — inclusivo ma elimina gli atomi completamente
    sepolti.
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np

from .feature_extraction import AtomFeature

try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.SASA import ShrakeRupley
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False


def compute_atom_sasa(pdb_path: str) -> Dict[tuple, float]:
    """
    Calcola il SASA per ogni atomo della struttura PDB.

    Usa l'algoritmo di Shrake & Rupley (1973) implementato in BioPython.
    Il raggio della sonda è 1.4 Å (molecola d'acqua standard).

    Returns
    -------
    dict  {(chain_id, res_seq, atom_name): sasa_value_angstrom2}
    """
    if not _HAS_BIOPYTHON:
        raise ImportError(
            "BioPython non installato. Esegui: pip install biopython"
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    sr = ShrakeRupley()
    sr.compute(structure, level="A")   # level="A" → per atomo

    sasa_map = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_seq = residue.get_id()[1]
                chain_id = chain.get_id()
                for atom in residue:
                    key = (chain_id, res_seq, atom.get_name().strip())
                    sasa_map[key] = atom.sasa

    return sasa_map


def filter_surface_features(
    features: List[AtomFeature],
    mol,
    chain_id: str,
    res_seq: int,
    sasa_map: Dict[tuple, float],
    sasa_threshold: float = 1.0,
) -> List[AtomFeature]:
    """
    Filtra le feature farmacofore tenendo solo quelle con almeno un atomo
    esposto al solvente (SASA > soglia).

    Parameters
    ----------
    features : List[AtomFeature]
        Feature estratte dallo Step 2 per UN residuo.
    mol : RDKit Mol
        Molecola RDKit del residuo (per recuperare i nomi degli atomi).
    chain_id : str
        Catena del residuo (es. "A").
    res_seq : int
        Numero sequenziale del residuo nel PDB.
    sasa_map : dict
        Mappa {(chain_id, res_seq, atom_name): sasa} calcolata su tutta la
        struttura da compute_atom_sasa().
    sasa_threshold : float
        Soglia minima di SASA in Å² per considerare un atomo esposto.
        Default 1.0 Å² — elimina solo gli atomi completamente sepolti.

    Returns
    -------
    List[AtomFeature]
        Sottoinsieme delle feature con esposizione sufficiente.
    """
    from rdkit.Chem import RemoveHs

    mol_no_h = RemoveHs(mol)
    atoms = list(mol_no_h.GetAtoms())

    filtered = []
    for feat in features:
        exposed = False
        for atom_idx in feat.atom_indices:
            if atom_idx >= len(atoms):
                continue
            atom_name = atoms[atom_idx].GetMonomerInfo()
            if atom_name is not None:
                name = atom_name.GetName().strip()
            else:
                # fallback: usa il simbolo dell'elemento
                name = atoms[atom_idx].GetSymbol()

            key = (chain_id, res_seq, name)
            sasa_val = sasa_map.get(key, 0.0)
            if sasa_val > sasa_threshold:
                exposed = True
                break

        if exposed:
            filtered.append(feat)

    return filtered


def filter_surface_features_by_coords(
    features: List[AtomFeature],
    sasa_map: Dict[tuple, float],
    chain_id: str,
    res_seq: int,
    all_atom_coords: List[dict],
    sasa_threshold: float = 1.0,
) -> List[AtomFeature]:
    """
    Versione alternativa che usa le coordinate atomiche del PDB per il matching,
    invece degli indici RDKit. Più robusta quando il mapping indice→nome è incerto.

    Ogni feature ha coords 3D; si trova l'atomo PDB più vicino e si controlla
    il suo SASA.

    Parameters
    ----------
    all_atom_coords : List[dict]
        Lista di {name, x, y, z} per gli atomi del residuo (da ResidueRecord.atoms).
    """
    if not all_atom_coords:
        return features

    atom_positions = np.array([[a["x"], a["y"], a["z"]] for a in all_atom_coords])
    atom_names = [a["name"].strip() for a in all_atom_coords]

    filtered = []
    for feat in features:
        # trova l'atomo PDB più vicino al centroide della feature
        dists = np.linalg.norm(atom_positions - feat.coords, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_name = atom_names[nearest_idx]

        key = (chain_id, res_seq, nearest_name)
        sasa_val = sasa_map.get(key, 0.0)

        if sasa_val > sasa_threshold:
            filtered.append(feat)

    return filtered