import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.fid_susc_alg.fid_susc import ResolventSusceptibility
from tqdm import tqdm
from typing import Dict, List


def build_tfim_operators(L, h, J=1.0, periodic_boundary=False):
    """
    Builds the TFIM Hamiltonian and driving operator H_I.
    """
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    I2 = np.eye(2)

    def apply_local_op(op, site):
        res = 1
        for i in range(L):
            res = np.kron(res, op if i == site else I2)
        return res

    H_ZZ = np.zeros((2**L, 2**L))
    for i in range(L - 1):
        H_ZZ -= J * apply_local_op(sz, i) @ apply_local_op(sz, i + 1)

    if periodic_boundary:
        H_ZZ -= J * apply_local_op(sz, L - 1) @ apply_local_op(sz, 0)

    H_X = np.zeros((2**L, 2**L))
    for i in range(L):
        H_X -= apply_local_op(sx, i)

    # Base Hamiltonian
    H = H_ZZ + h * H_X

    # Since H(h + delta) = H_ZZ + (h + delta)*H_X
    # The derivative (H_I) is exactly H_X
    H_I = H_X

    return H, H_I


if __name__ == "__main__":
    L = 8
    # Sweep across the transverse field to map the QPT
    h_values = np.linspace(0.1, 2.0, 30)

    test_degrees = [1, 2, 3, 5, 10, 50]

    exact_results = []
    qsvt_results: Dict[int, List[float]] = {deg: [] for deg in test_degrees}

    print(f"Scanning QPT for TFIM with {L} spins..")

    for h in tqdm(h_values, desc="Scanning h", unit="h"):
        # 1. Build the system
        H, H_I = build_tfim_operators(L, h)

        # 2. Initialize our Resolvent class
        system = ResolventSusceptibility(H, H_I)

        # 3. Calculate Exact
        # Note: compute_exact() returns (chi_F, QFI). We only plot chi_F here.
        chi_exact, qfi_exact = system.compute_exact()
        exact_results.append(chi_exact / L)

        # 4. Calculate QSVT Approximations
        for deg in test_degrees:
            chi_approx, qfi_approx = system.compute_qsvt(degree=deg)
            qsvt_results[deg].append(chi_approx / L)

    # --- Plotting Module ---
    sns.set_theme(style="whitegrid")
    # Using the beautiful Set2 palette you requested
    palette = sns.color_palette("Set2")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Exact
    ax.plot(
        h_values,
        exact_results,
        color="k",
        linestyle="--",
        linewidth=2,
        label="Exact Theoretical",
    )

    # Plot QSVT
    for i, deg in enumerate(test_degrees):
        ax.plot(
            h_values,
            qsvt_results[deg],
            color=palette[i],
            marker="o",
            markersize=5,
            linestyle="-",
            linewidth=2,
            label=f"QSVT Degree = {deg}",
            alpha=0.75,
        )

    # Theoretical infinite-size critical point
    ax.axvline(x=1.0, color="grey", linestyle=":", label="L→∞ Critical Point (h/J=1)")

    ax.set_title(f"TFIM Quantum Phase Transition: Exact vs. QSVT (L={L})", fontsize=14)
    ax.set_xlabel("Transverse Field Strength (h/J)", fontsize=12)
    ax.set_ylabel("Normalized Fidelity Susceptibility ($\\chi_F / L$)", fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()
