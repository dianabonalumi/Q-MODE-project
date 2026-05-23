# Dataset

Questa directory contiene i dataset forniti dal professore.

## Formato atteso

**CSV** (`dataset.csv`):
```
smiles,name
CC(N)C(=O)O,ALA
NCC(=O)O,GLY
...
```

**SDF** (`structures.sdf`): file standard con strutture 2D o 3D.

## Come usarli

```bash
python scripts/run_dataset.py --input data/raw/dataset.csv --output data/processed/
```
