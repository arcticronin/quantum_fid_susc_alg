import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.fid_susc_alg.model import build_tfim_operators
from src.fid_susc_alg.spectral import compute_dynamical_susceptibility


if __name__ == "__main__":
    L = 7
    # Test three distinct phases: Ferromagnetic, Critical, Paramagnetic
    h_test_values = [0.2, 0.4, 0.8, 0.9]
    frequencies = np.linspace(0, 6, 400)  # 0 6 400

    # --- Plotting Setup ---
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2")
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Run the Experiment Loop ---
    for idx, h_val in enumerate(h_test_values):
        print(f"Computing frequency-domain response for L={L} at h={h_val}...")

        H, H_I = build_tfim_operators(L, h_val)
        spectral_function = compute_dynamical_susceptibility(
            H, H_I, frequencies, eta=0.05
        )

        # Plot each phase with a different color from the palette
        ax.plot(
            frequencies,
            spectral_function,
            color=palette[idx],
            linewidth=2,
            label=f"h/J = {h_val}",
            alpha=0.75,
        )
        ax.fill_between(frequencies, spectral_function, color=palette[idx], alpha=0.15)

    # --- Formatting ---
    ax.set_title(f"Tracking the Spectral Gap Closing (TFIM, L={L})", fontsize=14)
    ax.set_xlabel("Frequency $\\omega$", fontsize=12)
    ax.set_ylabel("Spectral Response $\\chi''(\\omega)$", fontsize=12)
    ax.legend(title="Transverse Field", fontsize=11, title_fontsize=12)

    # Add text explaining the physics
    ax.text(
        3.5,
        ax.get_ylim()[1] * 0.75,
        "The lowest-frequency peak\n(the spectral gap) moves closest to\n$\\omega=0$ exactly at the (theoretical) critical point $h=1.0$.",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.show()
