"""
Downloads a few real PDB structures and, for each, crops a binding pocket as
the protein residues with at least one atom within CUTOFF A of a ligand
(non-solvent HETATM) atom. Pockets are saved to data/raw/.

Uso:
    python scripts/make_pockets.py
"""

import os
import sys
import urllib.request

CUTOFF = 5.0   # A: ligand-residue distance defining the pocket

# (PDB id, ligand code) -- deliberately varied targets
TARGETS = [
    ("3PTB", "BEN"),   # trypsin + benzamidine (charged/polar site)
    ("1HSG", "MK1"),   # HIV-1 protease + indinavir (extended hydrophobic site)
    ("4DFR", "MTX"),   # dihydrofolate reductase + methotrexate (mixed)
    ("1STP", "BTN"),   # streptavidin + biotin (H-bond network)
]

# Additional targets from the Astex Diverse Set (Hartshorn et al.), a curated
# benchmark of clean single-ligand crystal complexes -- used here to broaden
# the validation set for the Grover interactivity/distance ranking.
ASTEX_TARGETS = [
    ("1MMV", "3AR"), ("1L2S", "STC"), ("1N46", "PFA"), ("1Y6B", "AAX"),
    ("1YWR", "LI9"), ("1YV3", "BIT"), ("1R9O", "FLP"), ("1IG3", "VIB"),
    ("1R58", "AO5"), ("1YGC", "905"), ("1GPK", "HUP"), ("1HNN", "SKF"),
    ("1K3U", "IAD"), ("1Q1G", "MTI"), ("1GM8", "SOX"), ("1OPK", "P16"),
    ("1U1C", "BAU"), ("1T46", "STI"), ("1HQ2", "PH2"), ("1N2V", "BDI"),
    ("1S19", "MC9"), ("1OYT", "FSN"), ("1XM6", "5RM"), ("1TZ8", "DES"),
    ("2BSM", "BSM"),
]
TARGETS += ASTEX_TARGETS

# hetero groups that are never real ligands (solvent, ions, common cryoprotectants)
NON_LIGAND = {
    "HOH", "WAT", "TIP3", "DOD",
    "NA", "K", "CL", "MG", "CA", "ZN", "MN", "FE", "SO4", "PO4",
    "GOL", "EDO", "PEG", "ACT", "DMS", "MPD", "FMT",
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_pdb(pdb_id: str) -> str:
    path = os.path.join(RAW_DIR, f"{pdb_id}.pdb")
    if os.path.exists(path):
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  scarico {pdb_id} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, path)
    print("ok")
    return path


def parse_coord(line):
    return (
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54]),
    )


def extract_pocket(pdb_path: str, ligand_code: str, out_path: str) -> int:
    """Crops ATOM residues within CUTOFF of the given ligand's atoms."""
    atom_lines = []          # all protein ATOM lines
    ligand_atoms = []        # coordinates of the chosen ligand's atoms

    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            res_name = line[17:20].strip()
            if rec == "ATOM":
                atom_lines.append(line)
            elif rec == "HETATM" and res_name == ligand_code:
                try:
                    ligand_atoms.append(parse_coord(line))
                except ValueError:
                    pass

    if not ligand_atoms:
        print(f"  [!] ligando {ligand_code} non trovato in {os.path.basename(pdb_path)}")
        return 0

    # residues (chain, seq) with an atom within CUTOFF of the ligand
    cutoff_sq = CUTOFF ** 2
    pocket_keys = set()
    for line in atom_lines:
        try:
            x, y, z = parse_coord(line)
        except ValueError:
            continue
        for lx, ly, lz in ligand_atoms:
            if (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2 <= cutoff_sq:
                key = (line[21], line[22:26])   # chain + resseq
                pocket_keys.add(key)
                break

    # write all ATOM lines for the selected residues
    n_atoms = 0
    with open(out_path, "w") as out:
        out.write(f"HEADER    POCKET OF {os.path.basename(pdb_path)} AROUND {ligand_code}\n")
        for line in atom_lines:
            key = (line[21], line[22:26])
            if key in pocket_keys:
                out.write(line)
                n_atoms += 1
    return len(pocket_keys), n_atoms


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Estrazione tasche da strutture PDB reali\n" + "=" * 45)
    made = []
    for pdb_id, lig in TARGETS:
        try:
            src = download_pdb(pdb_id)
        except Exception as e:
            print(f"  [!] download {pdb_id} fallito: {e}")
            continue
        out = os.path.join(RAW_DIR, f"{pdb_id.lower()}_pocket.pdb")
        res = extract_pocket(src, lig, out)
        if res and res[0]:
            n_res, n_atoms = res
            print(f"  {pdb_id} ({lig}): tasca = {n_res} residui, {n_atoms} atomi → {out}")
            made.append(out)
    print(f"\nTasche create: {len(made)}")
    return made


if __name__ == "__main__":
    main()
