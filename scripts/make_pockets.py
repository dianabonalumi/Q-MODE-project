"""
Estrazione di tasche di legame da strutture PDB reali
=====================================================
Scarica alcune strutture dal PDB e, per ciascuna, ritaglia la TASCA come
l'insieme dei residui proteici con almeno un atomo entro CUTOFF Å da un atomo
del ligando (HETATM non-solvente). Le tasche risultanti vengono salvate in
data/raw/.

Uso:
    python scripts/make_pockets.py
"""

import os
import sys
import urllib.request

CUTOFF = 5.0   # Å: distanza ligando–residuo per definire la tasca

# (PDB id, codice del ligando di interesse) — bersagli diversi tra loro
TARGETS = [
    ("3PTB", "BEN"),   # tripsina + benzamidina (sito carico/polare)
    ("1HSG", "MK1"),   # HIV-1 protease + indinavir (sito idrofobico esteso)
    ("4DFR", "MTX"),   # diidrofolato reduttasi + methotrexate (misto)
    ("1STP", "BTN"),   # streptavidina + biotina (rete di legami H)
]

# Eteroatomi da NON considerare ligandi (solvente, ioni, crioprotettori comuni)
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
    """Ritaglia i residui ATOM entro CUTOFF da un atomo del ligando dato."""
    atom_lines = []          # tutte le righe ATOM (proteina)
    ligand_atoms = []        # coordinate degli atomi del ligando scelto

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

    # residui (chain, seq) con un atomo entro CUTOFF dal ligando
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

    # scrive tutte le righe ATOM dei residui selezionati
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
