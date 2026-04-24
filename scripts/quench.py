import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.fid_susc_alg.dqpt import simulate_dqpt


if __name__ == "__main__":
    L = 6
    h_init = 0.2  # Deep Ferromagnetic
    h_quench = 1.5  # Quench to Paramagnetic (crossing h=1.0)
    t_max = 5  # 4.0
    num_steps = 100  # 100

    print(f"Simulating Quench from h={h_init} to h={h_quench}...")
    times, le, qfi = simulate_dqpt(L, h_init, h_quench, t_max, num_steps, tqdm=tqdm)

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Plot: Loschmidt Echo
    ax1.plot(times, le, color=palette[0], linewidth=2, label="Loschmidt Echo $L(t)$")
    ax1.set_ylabel("Loschmidt Echo $L(t)$", fontsize=12)
    ax1.set_title(
        f"Dynamical Quantum Phase Transition (L={L}, $h_i={h_init} \\to h_f={h_quench}$)",
        fontsize=14,
    )
    ax1.legend(fontsize=11)

    # Bottom Plot: QFI
    ax2.plot(times, qfi, color=palette[1], linewidth=2, label="Normalized QFI")
    ax2.set_ylabel("Normalized QFI", fontsize=12)
    ax2.set_xlabel("Time (t/J)", fontsize=12)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.show()
