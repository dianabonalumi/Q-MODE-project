# Q-MODE — Report Tecnico

### Mappatura di tasche di legame proteiche su reticoli farmacoforici discreti per il docking quantistico

---

## Abstract

Q-MODE è una pipeline di pre-processing che traduce una **tasca di legame proteica** (fornita come file `.pdb`) in una **rappresentazione discreta, ordinata e numerica**, adatta sia al machine learning classico sia a un modello di docking quantistico. L'idea centrale è abbandonare le coordinate atomiche 3D continue — troppo dettagliate e variabili per essere mappate su qubit — a favore di un **reticolo intero 2D** in cui ogni nodo porta un'etichetta farmacoforica (idrofobico, donatore/accettore di legame idrogeno, aromatico, ionizzabile). Questa astrazione discreta è la condizione che rende il problema codificabile su un numero limitato di qubit.

Il progetto implementa la fase algoritmica di collegamento tra dato strutturale grezzo e codifica quantistica, fase lasciata a livello puramente teorico nella letteratura di riferimento (il framework dell'*inner lattice* e l'algoritmo quantistico per l'identificazione dei siti di docking nello spazio di interazione).

---

## 1. Contesto biologico

### 1.1 Proteine, amminoacidi e residui

Una proteina è una catena di amminoacidi. Quando due amminoacidi si uniscono formano un **legame peptidico**, con perdita di una molecola d'acqua: l'amminoacido all'interno della catena non è quindi più "completo" come da isolato. L'unità ripetuta che rimane lungo la catena si chiama **residuo**.

> **Residuo = un amminoacido così come appare quando è legato agli altri nella catena proteica.**

Ogni residuo è composto da due parti:

- **Backbone (scheletro)**: la sequenza ripetuta `N – Cα – C – O`, **identica in tutti i 20 amminoacidi**. Forma la spina dorsale della catena. Il carbonio alfa (Cα) è il punto di riferimento comune a tutti i residui.
- **Sidechain (catena laterale, "R")**: il gruppo attaccato al Cα, **diverso per ogni amminoacido**. È la sidechain che conferisce identità chimica: rende la leucina idrofobica, l'aspartato carico negativamente, la fenilalanina aromatica.

### 1.2 La tasca di legame

Ripiegandosi nello spazio, la catena crea cavità. La **tasca di legame** (binding pocket) è la cavità in cui un ligando (es. un farmaco) si inserisce e si lega. I residui che ne formano le pareti sono quelli rilevanti per il docking — e possono essere lontani nella sequenza primaria ma vicini nello spazio per effetto del ripiegamento. Nel dataset usato (`1a08`, HIV-1 protease) il file `1a08_pocket.pdb` contiene solo i residui che tappezzano la tasca, già ritagliati da uno strumento esterno.

### 1.3 Gruppi farmacoforici

Ragionare atomo per atomo è troppo dettagliato. Ciò che governa il legame non sono i singoli atomi ma i **modi in cui una regione può interagire** con un partner. Questi modi astratti sono le **feature farmacofore**.

> **Farmacoforo = l'insieme astratto delle proprietà chimico-fisiche che permettono a una molecola di legarsi a un'altra.**

Q-MODE usa sei categorie, l'alfabeto delle interazioni non covalenti:

| Tipo | Interazione fisica | Esempio |
|---|---|---|
| **HBondDonor** | Dona un idrogeno in legame H | –NH, –OH (serina) |
| **HBondAcceptor** | Accetta un idrogeno (doppietti liberi) | C=O carbonilico, N |
| **Hydrophobe** | Effetto idrofobico (desolvatazione) | catene di C (leucina, valina) |
| **Aromatic** | π-stacking | anelli (fenilalanina, triptofano) |
| **PosIonizable** | carica positiva | lisina, arginina |
| **NegIonizable** | carica negativa | aspartato, glutammato |

**Relazione gerarchica chiave**: un *residuo genera uno o più farmacofori*; non sono i farmacofori a essere fatti di residui. Più residui formano una tasca; un singolo residuo emette più farmacofori (proprietà interne alla sua chimica). Un'isoleucina genera vari siti idrofobici (dalla sidechain) più siti di legame H (dal backbone); una glicina, priva di sidechain reattiva, ne genera pochissimi.

---

## 2. Motivazione computazionale

Il docking molecolare cerca la complementarità tra i farmacofori del ligando e quelli della tasca: dove la tasca ha un accettore di H, il ligando offre un donatore; dove è idrofobica, il ligando offre una parte grassa. Modellare questa complementarità su un computer quantistico richiede di:

1. **discretizzare** lo spazio continuo (qubit ⇒ stati finiti);
2. **ridurre la dimensionalità** (meno qubit possibile);
3. **ordinare** i siti in una sequenza, per poter far "scorrere" il ligando lungo la tasca (sliding window).

Q-MODE realizza esattamente questi tre obiettivi attraverso una catena di traduzioni:

```
Residui (biologia)
  → Farmacofori 3D (chimica delle interazioni)
    → Reticolo 2D discreto etichettato (informatica)
      → Sequenza 1D ordinata di coppie (idrofobicità, legame-H)
        → Stati e ampiezze di qubit (fisica quantistica)
```

---

## 3. Architettura della pipeline

La pipeline si articola in 6 step più la codifica quantistica finale. Ogni step corrisponde a un modulo del package `amino_lattice/`.

### Step 1 — Lettura del PDB e ricostruzione molecolare
*(`pdb_reader.py`)*

Un parser a colonne fisse legge il PDB e raggruppa gli atomi per residuo in oggetti `ResidueRecord`. Vengono mantenuti solo i 20 amminoacidi standard (più varianti di protonazione dell'istidina e la cisteina disolfuro); acqua, ioni e ligandi sono scartati: **la pipeline modella la sola proteina**.

Il problema cruciale: il PDB fornisce posizioni e nomi atomici, **non la topologia chimica** (legami, aromaticità, cariche), necessaria a RDKit per estrarre i farmacofori. La soluzione adottata (`residue_to_mol`):

1. costruisce la molecola dallo **SMILES canonico** del residuo → topologia corretta;
2. genera un conformero 3D con ETKDGv3 (seed fisso, riproducibile);
3. **sovrascrive** le coordinate del conformero con quelle **cristallografiche** del PDB.

Si combinano così connettività chimica garantita e geometria sperimentale.

I residui vengono infine ordinati per **distanza del Cα dal centroide della tasca**: i più centrali — i più probabili contatti col ligando — vengono per primi. Questo ordine geometrico diventa l'asse della sequenza finale.

### Step 2 — Estrazione delle feature farmacofore
*(`feature_extraction.py`)*

Da ogni residuo si estraggono le `AtomFeature` (tipo + coordinate 3D + intensità) da due sorgenti:

- **Feature factory di RDKit** (`BaseFeatures.fdef`): riconosce i pattern farmacoforici via SMARTS. Per gruppi multi-atomo (es. anello aromatico) la feature è localizzata nel **centroide** degli atomi coinvolti.
- **Idrofobicità via contributi di Crippen**: il LogP è scomposto in contributi atomici; gli atomi con contributo positivo diventano siti `Hydrophobe` con **intensità = LogP parziale**. L'idrofobicità è così trattata come grandezza continua e per-atomo — la quantità `h` che la parte quantistica userà.

### Step 3 — Scelta di K e selezione dei siti rappresentativi
*(`site_selection.py`)*

Ogni residuo viene ridotto a **K siti**, dove K non è fisso ma scelto per residuo da `choose_k`:

- `active_features` (default): K = numero di farmacofori distinti (fusione di quelli dello stesso tipo a distanza < 1.5 Å);
- `heavy_atoms`, `fixed`, `groups` (elbow penalizzato BIC-like) come alternative.

Poi `select_representative_sites` esegue un **K-Means** sulle coordinate 3D dei farmacofori grezzi: ogni cluster → un sito, con coordinate = centroide, tipo = tipo dominante, intensità = somma. È una compressione: i molti farmacofori ridondanti di un residuo (es. più carboni idrofobici adiacenti) collassano in K rappresentanti puliti.

Infine i K siti vengono **riordinati secondo la topologia covalente**: una BFS sul grafo dei legami RDKit, partendo dall'atomo di indice minore (convenzionalmente l'N-terminale), con fallback sull'asse principale PCA. Così la sequenza dei siti segue la connettività chimica della molecola.

*(Nota: il K-Means è un hard clustering — ogni farmacoforo grezzo appartiene a un solo sito; due siti non condividono atomi.)*

### Step 4 — Proiezione geometrica 3D → 2D
*(`lattice_fitting.py`)*

I K siti 3D vengono schiacciati su un piano 2D (il reticolo è bidimensionale per minimizzare i qubit):

- **PCA** (default): tiene le 2 componenti principali, preserva la struttura globale;
- **MDS metrico**: preserva meglio le distanze a coppie per nuvole non planari.

Le coordinate sono poi scalate dividendo per `lattice_spacing` (Å per passo di reticolo, default 1.5) e centrate sul baricentro. La qualità della proiezione è misurata dallo **stress** di Kruskal (`< 0.1` = buono).

### Step 5 — Snapping ai nodi interi
*(`snapping.py`)*

Le coordinate 2D continue vengono discretizzate in nodi interi `(i,j)`. Per evitare che due siti collidano sullo stesso nodo, la strategia `hungarian` (default) genera nodi candidati attorno ai siti e risolve un problema di **assegnazione ottima** (algoritmo ungherese) che minimizza lo spostamento totale **garantendo nodi distinti**. Il raggio dei candidati è adattivo e ci sono `assert` che falliscono rumorosamente in caso di collisione residua.

### Step 6 — Etichettatura farmacoforica
*(`labeling.py`)*

Il tipo di ciascun sito è convertito in vettore numerico — `one_hot` (default, lunghezza 6), `index`, o `embedding` denso — producendo i `LabeledSite` finali `(i, j, tipo, vettore)`. La funzione `encode_chain` impacchetta i siti in una matrice `(K, 2 + label_dim)` pronta per il ML classico (le prime due colonne sono le coordinate del reticolo).

### Step 7 — Codifica quantistica
*(`qubit_chain.py`, `quantum_encoding.py`)*

Sulla **sequenza flat 1D** di tutti i siti di tutti i residui (vedi §4) agisce una **sliding window** lunga `ligand_size`: il ligando "scorre" lungo la tasca, e per ogni posizione si genera un segmento. Ogni sito è prima ridotto a due scalari `(h, hb)` (idrofobicità, legame H). Poi:

- **First encoding** (per Grover search): 2 bit per sito (h e hb sopra/sotto soglia a metà range) → stato di base, es. `|100111⟩`;
- **Second encoding** (per la distanza euclidea via interferenza): 4 ampiezze normalizzate `(a,b,c,d)` che codificano i valori continui di h/hb come angoli di stato.

---

## 4. Dal reticolo 2D al vettore 1D

Il passaggio 2D → 1D **non** comprime le coordinate `(i,j)` in un numero: è una **serializzazione**, cioè la lettura ordinata dei siti. Avviene con due ordinamenti annidati:

1. **tra residui** — ordinati per distanza dal centroide (il più centrale per primo);
2. **dentro ogni residuo** — i K siti ordinati per topologia covalente (o PCA in fallback).

Concatenando, si ottiene la `flat_chain`: una fila `[sito_0, sito_1, …]` in cui l'indice di posizione nasce dall'ordine di lettura, non dalle coordinate.

**Sottigliezza importante**: nella parte quantistica le coordinate `(i,j)` vengono **scartate**. Ciò che sopravvive in 1D è solo l'ordine dei siti e, per ciascuno, la coppia `(h, hb)` derivata da tipo + intensità. Il reticolo 2D serve a *imporre l'ordine e la struttura geometrica* (ed è centrale per la visualizzazione e per l'ML classico, dove invece `(i,j)` è conservato in `encode_chain`), ma il canale che raggiunge i qubit è la sequenza ordinata di `(h, hb)`.

---

## 5. Strutture dati e percorsi di esecuzione

**Dataclass portanti**: `AtomFeature` (farmacoforo/sito 3D), `LabeledSite` (sito finale su reticolo), `ResidueRecord` (residuo dal PDB), `MappingResult` (risultato per amminoacido).

**Due punti di ingresso**:

- **`scripts/run_pocket.py`** — orchestratore di produzione: PDB intero → cicla su tutti i residui (reticolo locale indipendente per ciascuno) → sequenza flat → encoding quantistico → output JSON/CSV + 3 figure. È il percorso documentato e usato per i risultati.
- **`AminoLatticePipeline`** — motore per singolo amminoacido (da SMILES o `Mol`), restituisce un `MappingResult` ricco. Utile per esperimenti, test e ML su residui isolati.

Concettualmente l'orchestratore applica il motore a ogni residuo; le due implementazioni condividono i sei step.

---

## 6. Esempio illustrativo

Sull'esempio `1a08_pocket.pdb` (38 residui della tasca della HIV-1 protease) la pipeline produce **203 siti** in sequenza. I primi siti, partendo dalla tirosina più centrale:

```
pos 0: A205_TYR (i=3,  j=0)  HBondAcceptor  intensity=2.92
pos 1: A205_TYR (i=1,  j=1)  Hydrophobe     intensity=0.32
pos 2: A205_TYR (i=1, j=-1)  Hydrophobe     intensity=0.32
pos 3: A205_TYR (i=0,  j=0)  Hydrophobe     intensity=0.14
pos 4: A205_TYR (i=-2, j=1)  HBondAcceptor  intensity=6.76
pos 5: A205_TYR (i=-3,j=-1)  HBondAcceptor  intensity=3.11
pos 6: A189_LEU (i=2,  j=1)  Hydrophobe     intensity=0.14
...
```

Si osserva la struttura attesa: la tirosina (aromatica con –OH) genera un mix di siti idrofobici e accettori di H; la leucina (alifatica) genera siti puramente idrofobici. Le coordinate `(i,j)` sono locali al residuo (per questo si ripetono tra residui diversi).

---

## 7. Output

- `pocket_chain.json` / `.csv`: la sequenza flat completa con metadati per sito (residuo, i, j, tipo, intensità);
- `quantum_chain.json`: i segmenti della sliding window con stato di first-encoding e ampiezze di second-encoding;
- tre figure: griglia dei reticoli locali, heatmap tipo × posizione, barchart di composizione farmacoforica per residuo.

---

## 8. Limiti e punti aperti

1. **Intensità del legame idrogeno = placeholder.** L'intensità HB è una funzione pseudo-casuale ma deterministica delle coordinate, non un'energia fisica. L'asse `hb` dell'encoding quantistico non ha quindi fondamento fisico finché non sostituito con un modello geometrico (distanze/angoli donatore-accettore).
2. **Overlay delle coordinate euristico.** Il mapping coordinate-PDB → atomi-RDKit avviene per ordine posizionale, non per identità chimica: in molti residui può assegnare coordinate non corrette ai singoli atomi. Gli step a valle (clustering, PCA) ne attenuano l'effetto, ma è il primo punto da irrobustire per precisione geometrica.
3. **Parte quantistica = solo encoding.** Sono implementate le codifiche (binaria e in ampiezza) ma non l'esecuzione degli algoritmi quantistici (Grover, swap test); non c'è quindi uno scoring di docking calcolato.
4. **Assenza di validazione quantitativa.** Un solo PDB di esempio, nessun benchmark o confronto con pose/score di riferimento.
5. **Reticolo locale, non globale.** Le coordinate `(i,j)` sono per-residuo; manca un sistema di riferimento condiviso tra residui (scelta deliberata, ma limita le relazioni geometriche inter-residuo).

*(Nota di manutenzione: una duplicazione di codice in `site_selection.py` — ora rimossa — disattivava l'ordinamento topologico e rompeva il percorso `AminoLatticePipeline`; la correzione ha ripristinato entrambi.)*

---

## 9. Conclusioni

Q-MODE implementa in modo modulare e leggibile l'intera catena che porta da una tasca proteica grezza a una rappresentazione discreta pronta per la codifica quantistica, colmando un vuoto algoritmico della letteratura di riferimento. I sei step sono ben separati, deterministici e testati a livello unitario; la riduzione progressiva (residui → farmacofori → K siti → reticolo 2D → sequenza 1D → qubit) è coerente e motivata a ogni passaggio. I limiti principali riguardano il realismo fisico di alcune grandezze (intensità HB), la robustezza del mapping geometrico e l'assenza di una validazione quantitativa e dell'esecuzione effettiva della parte quantistica — naturali sviluppi futuri.
