import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmode.grover import tile_offset, search_docking_sites

H_THR = 1.0
HB_THR = 1.0


def _site(residue, feature_type, intensity):
    return {"residue": residue, "type": feature_type, "intensity": intensity}


# Un sito reale ha sempre un canale a zero (get_h_hb_intensities), quindi "11"
# per sito non è raggiungibile dalla proteina — solo dal ligando.
FLAT_CHAIN = [
    _site("R1", "Hydrophobe", 2.0),    # "10"
    _site("R1", "HBondDonor", 2.0),    # "01"
    _site("R2", "Hydrophobe", 2.0),    # "10"
    _site("R2", "Hydrophobe", 2.0),    # "10"
    _site("R3", "Hydrophobe", 0.5),    # "00" (sotto soglia)
    _site("R3", "HBondDonor", 0.5),    # "00" (sotto soglia)
    _site("R4", "HBondDonor", 2.0),    # "01"
    _site("R4", "HBondDonor", 2.0),    # "01"
]


def test_tile_offset_windows_and_bitstrings():
    unique, latest = tile_offset(FLAT_CHAIN, ligand_size=2, offset=0, h_thr=H_THR, hb_thr=HB_THR)
    # [0-1]->"10"+"01"="1001"; [2-3]->"1010"; [4-5]->"0000"; [6-7]->"0101"
    assert unique == ["1001", "1010", "0000", "0101"]
    assert latest == {"1001": 0, "1010": 2, "0000": 4, "0101": 6}


def test_tile_offset_offset_one_drops_partial_windows():
    # offset=1 su 8 siti, ligand_size=2 -> finestre [1-2],[3-4],[5-6]; sito 7 scartato (parziale)
    unique, latest = tile_offset(FLAT_CHAIN, ligand_size=2, offset=1, h_thr=H_THR, hb_thr=HB_THR)
    assert len(unique) == 3
    assert 7 not in latest.values()


def test_search_finds_exact_match():
    # Il ligando corrisponde esattamente alla finestra [2-3] (bitstring "1010")
    ligand_hbs = [(2.0, 0.0), (2.0, 0.0)]
    candidates = search_docking_sites(FLAT_CHAIN, ligand_hbs, ligand_size=2,
                                       h_thr=H_THR, hb_thr=HB_THR, shots=4096)
    assert any(c["window_start_index"] == 2 and c["shift_offset"] == 0 for c in candidates)
    for c in candidates:
        assert 0.0 <= c["matching_probability"] <= 1.0
        assert c["matching_probability"] >= c["threshold"]


def test_search_no_match_when_pattern_absent():
    # "1111" non può comparire in nessuna finestra: i siti della proteina non
    # producono mai il bit "11" per singolo sito (vedi commento sopra), quindi
    # nessuna finestra a 2 siti può risultare "1111".
    ligand_bitstring = "1111"
    assert ligand_bitstring not in tile_offset(FLAT_CHAIN, 2, 0, H_THR, HB_THR)[0]
    assert ligand_bitstring not in tile_offset(FLAT_CHAIN, 2, 1, H_THR, HB_THR)[0]

    ligand_hbs = [(2.0, 2.0), (2.0, 2.0)]  # ogni sito -> "11" (solo il ligando può)
    candidates = search_docking_sites(FLAT_CHAIN, ligand_hbs, ligand_size=2,
                                       h_thr=H_THR, hb_thr=HB_THR, shots=4096)
    assert candidates == []


def test_ligand_size_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        search_docking_sites(FLAT_CHAIN, [(1.0, 1.0)], ligand_size=2,
                              h_thr=H_THR, hb_thr=HB_THR)
