def formula_bin_specific(bin_id, L, N, q):
    return {'bright': 0.48, 'medium': 0.42, 'dark': 0.35}.get(bin_id, 0.42)
