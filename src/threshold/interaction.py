def formula_interaction(bin_id, L, N, q, gamma=0.25, tau_floor=0.30, tau_base=0.48):
    return tau_base * (1 - gamma*(1 - L)*N) * q + tau_floor*(1 - q)
