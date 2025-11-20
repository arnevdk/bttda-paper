import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from mne.time_frequency import read_tfrs
from mne.viz import plot_topomap

axis_width = 0.25
plt.rcParams.update(
    {
        "text.usetex": True,
        "axes.linewidth": axis_width,
        "ytick.major.width": axis_width,
    }
)

contrast_grand_avg = read_tfrs(
    "data/interpretability/mi_contrast_grand-avg_tfr.h5", verbose=False
)
contrast_b1 = read_tfrs(
    "data/interpretability/mi_contrast_block-1_tfr.h5", verbose=False
)
contrast_b2 = read_tfrs(
    "data/interpretability/mi_contrast_block-2_tfr.h5", verbose=False
)

contrast_rec = mne.combine_evoked([contrast_b1, contrast_b2], weights=[1, 1])


def viz_tfr(evoked, name, chan="C3", t=1, f=10):
    figs = evoked.copy().pick(chan).plot(show=False, colorbar=False)
    figs[0].gca().set_axis_off()
    plt.savefig(
        f"figures/interpretability/mi_contrast_{name}_spec.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True,
    )

    sfreq = len(evoked.times) / np.max(evoked.times)
    s = int((t - evoked.tmin) * sfreq)
    fs = np.where(evoked.freqs > f)[0][0]
    topo = evoked.data[:, fs, s]
    mask = np.array([c == chan for c in evoked.ch_names])
    plot_topomap(topo, evoked.info, show=False, mask=mask)

    plt.savefig(
        f"figures/interpretability/mi_contrast_{name}_topo.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True,
    )


viz_tfr(contrast_grand_avg, "grand-avg", chan="C3", t=1, f=11)
viz_tfr(contrast_b1, "block-1", chan="C3", t=1, f=11)
viz_tfr(contrast_b2, "block-2", chan="P4", t=1.6, f=10)
viz_tfr(contrast_rec, "rec", chan="C3", t=1, f=11)
