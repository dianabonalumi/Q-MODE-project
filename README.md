# Amino Acid Lattice Mapping — Pocket Edition

Pipeline di pre-processing per tradurre la **tasca di legame** di una proteina
(file `.pdb`) in una **catena di siti di interazione** su un reticolo 2D discreto.

Implementa i passi di pre-processing omessi in Lilliapoulos et al., che introduce
il concetto di *inner lattice* solo a livello teorico senza fornire l'algoritmo.

---

## Setup (una sola volta)

**Requisiti:** Python ≥ 3.9, pip.

```bash
# 1. Entra nella cartella
cd amino-lattice-mapping

# 2. Crea ambiente virtuale
python -m venv venv

# 3. Attivalo
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

# 4. Installa le dipendenze
pip install -e .
```

---

## Utilizzo

Metti il tuo file PDB in `data/raw/`, poi:

```bash
# Output testuale + plot interattivo
python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb --plot

# Salva JSON, CSV e immagine del plot
python scripts/run_pocket.py --pdb data/raw/1a08_pocket.pdb \
    --output data/processed/ \
    --save-plot data/processed/pocket.png
```

### Opzioni principali

| Opzione | Default | Descrizione |
|---|---|---|
| `--pdb` | (obbligatorio) | Percorso al file PDB della tasca o della proteina |
| `--output` | None | Directory dove salvare JSON e CSV |
| `--plot` | False | Apre la visualizzazione grafica interattiva |
| `--save-plot` | None | Salva il plot come PNG |
| `--projection` | `pca` | Metodo di proiezione 3D→2D: `pca` o `mds` |
| `--max-k` | 6 | Numero massimo di siti per residuo |
| `--k-strategy` | `active_features` | Come si sceglie K: `active_features`, `heavy_atoms`, `fixed`, `groups` |
| `--label-mode` | `one_hot` | Encoding del tipo farmacoforo: `one_hot`, `index`, `embedding` |

---

## Output

**Nel terminale:** tabella con un sito per riga — indice, residuo di origine, coordinate (i,j) sul reticolo, tipo farmacoforo.

**`data/processed/pocket_chain.json`** — catena completa con metadati:
```json
[{"site_idx": 0, "residue": "A156_ILE", "i": 2, "j": -3, "type": "Hydrophobe"}, ...]
```

**`data/processed/pocket_chain.csv`** — stessa informazione in formato tabellare per analisi ML.

---

## Struttura del codice

```
amino_lattice/
├── pdb_reader.py         # Legge il PDB → residui con coordinate 3D reali
├── feature_extraction.py # RDKit: estrae farmacofori, H-bond, idrofobicità
├── site_selection.py     # Sceglie K siti rappresentativi per residuo
├── lattice_fitting.py    # Proietta coordinate 3D → piano 2D (PCA/MDS)
├── snapping.py           # Coordinate continue → nodi interi (i,j)
├── labeling.py           # Tipo farmacoforo → vettore one-hot / embedding
└── pipeline.py           # Orchestratore per input da SMILES

scripts/
└── run_pocket.py         # Entry point principale per file PDB
```
