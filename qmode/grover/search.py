"""Grover search (Liliopoulos et al. 2025). SWAP-test evaluation/ranking non implementato."""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit_aer import AerSimulator

from qmode.quantum_encoding import first_encoding
from qmode.qubit_chain import get_h_hb_intensities


def tile_offset(
    flat_chain: List[dict],
    ligand_size: int,
    offset: int,
    h_thr: float,
    hb_thr: float,
) -> Tuple[List[str], Dict[str, int]]:
    """Finestre non sovrapposte di `ligand_size` siti da `offset` (Fig. 6-7).
    Ritorna i bitstring unici e, per ciascuno, l'indice della sua ultima occorrenza."""
    n_sites = len(flat_chain)
    latest_position: Dict[str, int] = {}
    order: List[str] = []

    start = offset
    while start + ligand_size <= n_sites:
        window = flat_chain[start:start + ligand_size]
        bitstring = "".join(
            first_encoding(*get_h_hb_intensities(site), h_thr, hb_thr)
            for site in window
        )
        if bitstring not in latest_position:
            order.append(bitstring)
        latest_position[bitstring] = start
        start += ligand_size

    return order, latest_position


def build_superposition(unique_bitstrings: List[str], n_qubits: int) -> np.ndarray:
    """|s⟩ = 1/√N Σ|x⟩ (Eq. 5): ampiezza 1/√N sui bitstring unici, zero altrove."""
    dim = 2 ** n_qubits
    s = np.zeros(dim, dtype=complex)
    n = len(unique_bitstrings)
    amp = 1.0 / np.sqrt(n)
    for bitstring in unique_bitstrings:
        s[int(bitstring, 2)] = amp
    return s


def build_oracle(ligand_bitstring: str, n_qubits: int) -> np.ndarray:
    """Ô = I - 2|x_i⟩⟨x_i| (Eq. 7)."""
    dim = 2 ** n_qubits
    oracle = np.eye(dim, dtype=complex)
    idx = int(ligand_bitstring, 2)
    oracle[idx, idx] = -1.0
    return oracle


def build_diffusion(s_vector: np.ndarray) -> np.ndarray:
    """Ĝ = 2|s⟩⟨s| - I (Eq. 8)."""
    dim = s_vector.shape[0]
    return 2.0 * np.outer(s_vector, s_vector.conj()) - np.eye(dim, dtype=complex)


def run_grover_circuit(
    s_vector: np.ndarray,
    oracle: np.ndarray,
    diffusion: np.ndarray,
    n_qubits: int,
    shots: int = 4096,
) -> Dict[str, float]:
    """StatePreparation(|s⟩) -> Oracolo -> Grover -> misura (Fig. 4).
    Ritorna {bitstring: probabilità}; get_counts() è già coerente con int(bitstring, 2),
    nessuna inversione di bit necessaria (verificato con uno stato di controllo noto)."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.append(StatePreparation(s_vector), range(n_qubits))
    qc.append(UnitaryGate(oracle), range(n_qubits))
    qc.append(UnitaryGate(diffusion), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    return {key: count / shots for key, count in counts.items()}


def search_docking_sites(
    flat_chain: List[dict],
    ligand_hbs: List[Tuple[float, float]],
    ligand_size: int,
    h_thr: float,
    hb_thr: float,
    shots: int = 4096,
) -> List[dict]:
    """Ricerca su ogni shift offset; ritorna le finestre con probabilità >= soglia 1/N."""
    if len(ligand_hbs) != ligand_size:
        raise ValueError(
            f"ligand_hbs deve avere {ligand_size} coppie (h, hb), trovate {len(ligand_hbs)}"
        )

    n_qubits = 2 * ligand_size
    ligand_bitstring = "".join(
        first_encoding(h, hb, h_thr, hb_thr) for h, hb in ligand_hbs
    )

    candidates: List[dict] = []

    for offset in range(ligand_size):
        unique_bitstrings, latest_position = tile_offset(
            flat_chain, ligand_size, offset, h_thr, hb_thr
        )
        if not unique_bitstrings:
            continue

        n = len(unique_bitstrings)
        threshold = 1.0 / n

        s_vector = build_superposition(unique_bitstrings, n_qubits)
        oracle = build_oracle(ligand_bitstring, n_qubits)
        diffusion = build_diffusion(s_vector)

        probabilities = run_grover_circuit(s_vector, oracle, diffusion, n_qubits, shots=shots)
        matching_probability = probabilities.get(ligand_bitstring, 0.0)

        if matching_probability >= threshold and ligand_bitstring in latest_position:
            window_start = latest_position[ligand_bitstring]
            window_sites = flat_chain[window_start:window_start + ligand_size]
            residues = list(dict.fromkeys(s["residue"] for s in window_sites))
            candidates.append({
                "shift_offset": offset,
                "window_start_index": window_start,
                "residues": residues,
                "ligand_bitstring": ligand_bitstring,
                "matching_probability": matching_probability,
                "threshold": threshold,
                "n_unique_states": n,
            })

    return candidates
