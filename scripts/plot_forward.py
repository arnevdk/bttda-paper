import pdb

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import plot_topomap

from setup import set_size, textwidth

plt.rcParams["axes.prop_cycle"] = "cycler('color', 'krb')"

for b in range(2):
    fig, axs = plt.subplots(
        1,
        4,
        layout="tight",
        sharex=False,
        width_ratios=[1, 1, 2, 2],
    )

    # Plot reconstructed contrast
    contrast_rec = mne.read_evokeds(f"data/forward/contrast_rec_block-{b}.fif")[0]
    contrast_rec.apply_baseline((None, 0))
    contrast_rec.nave = None

    contrast_rec.plot(axes=axs[3], show=False)
    axs[3].set_title("")
    axs[3].set_ylabel("Amplitude (µV)")
    axs[3].set_yticks([-2, 0, 2])
    axs[3].set_ylim([-2, 2])
    axs[3].set_xticks([-0.2, 0, 0.5, 1])
    axs[3].set_xlim([-0.2, 1])
    axs[3].axvline(0, linestyle="dotted")

    # Plot temporal aps
    aps_tmp = np.load(f"data/forward/ap_block-{b}_mode-{1}.npy")
    axs[2].plot(contrast_rec.times, aps_tmp)
    axs[2].set_yticks([-1, 0, 1])
    axs[2].set_ylim([-1.5, 1.5])
    axs[2].set_ylabel("Scale")
    axs[2].set_xticks([-0.2, 0, 0.5, 1])
    axs[2].set_xlabel("Time (s)")
    axs[2].set_xlim([-0.2, 1])
    axs[2].axvline(0, linestyle="dotted")

    # Plot spatial aps
    aps_sp = np.load(f"data/forward/ap_block-{b}_mode-{0}.npy")
    vmax = np.max(np.abs(aps_sp))
    for r in range(aps_sp.shape[-1]):
        plot_topomap(
            aps_sp[:, r],
            contrast_rec.info,
            axes=axs[r],
            show=False,
            cmap="turbo",
            vlim=(-vmax, vmax),
        )

    fig.set_size_inches(set_size(width_pt=textwidth, subplots=(1, 3)))
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(f"figures/forward_block-{b}.png", bbox_inches="tight", pad_inches=0)
    # fig.savefig(f"figures/forward_block-{b}.pgf", bbox_inches="tight", pad_inches=0)
    # fig.savefig(f"figures/forward_block-{b}.svg", bbox_inches="tight", pad_inches=0)
    # fig.savefig(f"figures/forward_block-{b}.pdf", bbox_inches="tight", pad_inches=0)
