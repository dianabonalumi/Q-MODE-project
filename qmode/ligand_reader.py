"""
Ligand Reader — da HETATM del PDB a molecola RDKit di un ligando generico.
Stesso schema di residue_to_mol() in pdb_reader.py: atomi reali dal PDB +
template noto per assegnare l'ordine dei legami. Il template qui viene dal
PDB Chemical Component Dictionary (CCD) via l'API REST di RCSB, invece del
dizionario fisso AMINO_SMILES usato per i 20 amminoacidi standard.

Se il codice del ligando non è nel CCD, o la rete non è disponibile, si
ricade su una bond-order perception geometrica (rdDetermineBonds), che non
richiede alcun template.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json
import urllib.request
import urllib.error
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem

from .pdb_reader import AMINO_SMILES, _is_hydrogen

CCD_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"
_CCD_CACHE: dict = {}


def fetch_ccd_smiles(ligand_code: str, timeout: float = 5.0) -> Optional[str]:
    """SMILES ideale del componente `ligand_code` dal PDB Chemical Component
    Dictionary. None su qualunque errore (rete, codice sconosciuto, campo
    mancante) — non solleva mai un'eccezione."""
    if ligand_code in _CCD_CACHE:
        return _CCD_CACHE[ligand_code]

    smiles = None
    try:
        url = CCD_URL.format(code=ligand_code.upper())
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
        descriptors = data.get("pdbx_chem_comp_descriptor", [])
        canonical = [d for d in descriptors if d.get("type") == "SMILES_CANONICAL"]
        any_smiles = [d for d in descriptors if "SMILES" in d.get("type", "")]
        chosen = (canonical or any_smiles or [None])[0]
        if chosen:
            smiles = chosen if isinstance(chosen, str) else chosen.get("descriptor")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        warnings.warn(f"fetch_ccd_smiles({ligand_code}): rete/parsing fallito ({e})")

    _CCD_CACHE[ligand_code] = smiles
    return smiles


@dataclass
class LigandRecord:
    res_name: str
    res_seq: int
    chain_id: str
    atoms: List[dict]
    mol: Optional[object] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"{self.chain_id}{self.res_seq}_{self.res_name}"


def load_ligand_from_pdb(
    pdb_path: str,
    ligand_code: Optional[str] = None,
    min_heavy_atoms: int = 5,
) -> Optional[LigandRecord]:
    """Legge le righe HETATM del PDB (esclude acqua e i 20 amminoacidi
    standard). Se ligand_code è dato, prende quel gruppo; altrimenti sceglie
    automaticamente il gruppo con più atomi pesanti (scarta ioni singoli
    sotto min_heavy_atoms)."""
    groups: dict = {}

    with open(pdb_path) as f:
        for line in f:
            record = line[:6].strip()
            if record != "HETATM":
                continue

            alt_loc  = line[16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq  = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            element  = line[76:78].strip() if len(line) > 76 else atom_name[0]

            if alt_loc and alt_loc != "A":
                continue
            if res_name in ("HOH", "WAT", "TIP3"):
                continue
            if res_name in AMINO_SMILES:
                continue
            if ligand_code and res_name != ligand_code.upper():
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            key = (chain_id, res_seq, res_name)
            if key not in groups:
                groups[key] = LigandRecord(
                    res_name=res_name, res_seq=res_seq, chain_id=chain_id, atoms=[]
                )
            groups[key].atoms.append({
                "name": atom_name, "x": x, "y": y, "z": z, "element": element
            })

    candidates = [
        rec for rec in groups.values()
        if sum(1 for a in rec.atoms if not _is_hydrogen(a)) >= min_heavy_atoms
    ]
    if not candidates:
        return None

    best = max(candidates, key=lambda rec: sum(1 for a in rec.atoms if not _is_hydrogen(a)))
    best.mol = ligand_to_mol(best)
    return best


def _ligand_atoms_to_pdb_block(rec: LigandRecord) -> str:
    lines = []
    i = 0
    for a in rec.atoms:
        if _is_hydrogen(a):
            continue
        i += 1
        name = a["name"].strip()
        elem = a.get("element", "").strip() or (name[0] if not name[0].isdigit() else name[1])
        lines.append(
            f"HETATM{i:5d} {name:<4s} {rec.res_name:<3s} {rec.chain_id:1s}{rec.res_seq:4d}    "
            f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}  1.00  0.00          {elem:>2s}"
        )
    return "\n".join(lines) + "\nEND\n"


def ligand_to_mol(rec: LigandRecord) -> Optional[object]:
    """Costruisce la molecola RDKit del ligando dagli atomi reali del PDB.
    Prova prima il template CCD (AssignBondOrdersFromTemplate, come per gli
    amminoacidi); se non disponibile ricade sulla bond-order perception
    geometrica (rdDetermineBonds)."""
    pdb_block = _ligand_atoms_to_pdb_block(rec)
    mol_from_pdb = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol_from_pdb is None:
        warnings.warn(f"{rec.label}: impossibile costruire la molecola dagli atomi PDB")
        return None

    smiles = fetch_ccd_smiles(rec.res_name)
    if smiles:
        template = Chem.MolFromSmiles(smiles)
        if template is not None:
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(template, mol_from_pdb)
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol, addCoords=True)
                return mol
            except Exception as e:
                warnings.warn(
                    f"{rec.label}: template CCD non applicabile ({e}), "
                    f"ricado sulla bond-order perception geometrica"
                )

    try:
        from rdkit.Chem import rdDetermineBonds
        mol = Chem.Mol(mol_from_pdb)
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol, addCoords=True)
        warnings.warn(f"{rec.label}: legami assegnati per geometria (nessun template CCD)")
        return mol
    except Exception as e:
        warnings.warn(f"{rec.label}: bond-order perception geometrica fallita ({e})")
        return None
