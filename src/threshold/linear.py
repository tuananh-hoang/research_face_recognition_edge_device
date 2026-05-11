def formula_linear(bin_id, L, N, q):
    tau_base = 0.48
    b = 0.10
    c = 0.05
    return tau_base - b*(1 - L) - c*N
