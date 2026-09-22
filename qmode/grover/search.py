"""Grover search (Liliopoulos et al. 2025). SWAP-test evaluation/ranking not implemented."""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit_aer import AerSimulator

from qmode.quantum_encoding import first_encoding
from qmode.qubit_chain import get_h_hb_intensities


def window_interactivity(window: List[dict]) -> float:
    """Aggregate interaction strength of a contiguous window: sum of each
    site's active h/hb intensity. Used to pick the most interactive window
    among several that collapse onto the same first-encoding bitstring."""
    return sum(sum(get_h_hb_intensities(site)) for site in window)


def tile_offset_ranked(
    flat_chain: List[dict],
    ligand_size: int,
    offset: int,
    h_thr: float,
    hb_thr: float,
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Non-overlapping windows of `ligand_size` sites starting at `offset`
    (Fig. 6-7). Returns the unique bitstrings and, for each, EVERY index
    that produces it, ordered by descending interactivity (sum of h/hb
    intensity across the window's sites), ties broken by position.

    The head of each list is what `tile_offset` returns; the tail is what
    the one-window-per-bitstring collapse throws away. Windows inside one
    offset are non-overlapping, so those extra positions are disjoint
    patches, not shifts of the same one."""
    n_sites = len(flat_chain)
    scored: Dict[str, List[Tuple[float, int]]] = {}
    order: List[str] = []

    start = offset
    while start + ligand_size <= n_sites:
        window = flat_chain[start:start + ligand_size]
        bitstring = "".join(
            first_encoding(*get_h_hb_intensities(site), h_thr, hb_thr)
            for site in window
        )
        if bitstring not in scored:
            order.append(bitstring)
            scored[bitstring] = []
        scored[bitstring].append((window_interactivity(window), start))
        start += ligand_size

    return order, {
        bitstring: [pos for _, pos in sorted(hits, key=lambda t: (-t[0], t[1]))]
        for bitstring, hits in scored.items()
    }


def tile_offset(
    flat_chain: List[dict],
    ligand_size: int,
    offset: int,
    h_thr: float,
    hb_thr: float,
) -> Tuple[List[str], Dict[str, int]]:
    """Non-overlapping windows of `ligand_size` sites starting at `offset`
    (Fig. 6-7). Returns the unique bitstrings and, for each, the index of
    its most interactive occurrence (highest sum of h/hb intensity across
    the window's sites) — not just whichever window was seen last."""
    order, positions = tile_offset_ranked(
        flat_chain, ligand_size, offset, h_thr, hb_thr
    )
    return order, {bitstring: pos[0] for bitstring, pos in positions.items()}


def build_superposition(unique_bitstrings: List[str], n_qubits: int) -> np.ndarray:
    """|s⟩ = 1/√N Σ|x⟩ (Eq. 5): amplitude 1/√N on each unique bitstring, zero elsewhere."""
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
    """StatePreparation(|s⟩) -> oracle -> Grover -> measure (Fig. 4).
    Returns {bitstring: probability}; get_counts() already lines up with
    int(bitstring, 2), no bit reversal needed (checked against a known
    control state).

    Scalability note: oracle and diffusion are dense unitary matrices
    compiled to a circuit via UnitaryGate; Qiskit's synthesis for this kind
    of matrix doesn't scale well past ~10 qubits (measured: ~45s per
    transpile at 12 qubits, repeated once per shift offset)."""
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
    per_offset: int = 1,
) -> List[dict]:
    """Searches every shift offset; returns windows with probability >= the
    1/N threshold, sorted by descending interactivity score (sum of h/hb
    intensity across the window's sites) so the most interactive matching
    window comes first.

    `per_offset` is how many matching windows each offset may contribute,
    the most interactive first (default 1, the historical behaviour). It
    widens the candidate set — up to `ligand_size * per_offset` windows —
    without touching the circuit: the superposition is built over the
    UNIQUE bitstrings of the tiling, so extra occurrences of the same
    bitstring are one entry there whatever this is set to."""
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
        unique_bitstrings, positions = tile_offset_ranked(
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

        if matching_probability >= threshold and ligand_bitstring in positions:
            for rank, window_start in enumerate(
                positions[ligand_bitstring][:per_offset], start=1
            ):
                window_sites = flat_chain[window_start:window_start + ligand_size]
                residues = list(dict.fromkeys(s["residue"] for s in window_sites))
                candidates.append({
                    "shift_offset": offset,
                    "rank_in_offset": rank,
                    "window_start_index": window_start,
                    "interactivity_score": round(window_interactivity(window_sites), 3),
                    "residues": residues,
                    "ligand_bitstring": ligand_bitstring,
                    "matching_probability": matching_probability,
                    "threshold": threshold,
                    "n_unique_states": n,
                })

    candidates.sort(key=lambda c: c["interactivity_score"], reverse=True)
    return candidates
