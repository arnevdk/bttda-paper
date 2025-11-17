import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne.viz import plot_topomap

contrast_grand_avg = mne.read_evokeds(
    "data/interpretability/erp_contrast_grand-avg_ave.fif", verbose=False
)[0]
contrast_b1 = mne.read_evokeds(
    "data/interpretability/erp_contrast_block-1_ave.fif", verbose=False
)[0]
contrast_b2 = mne.read_evokeds(
    "data/interpretability/erp_contrast_block-2_ave.fif", verbose=False
)[0]

contrast_rec = mne.combine_evoked([contrast_b1, contrast_b2], weights=[1, 1])


def save_evoked(evoked, name):
    df = pd.DataFrame(evoked.data.T * 1e6, index=evoked.times, columns=evoked.ch_names)
    df.index.name = "time"
    df.to_csv(f"data/interpretability/erp_{name}.csv")


save_evoked(contrast_grand_avg, "contrast_grand-avg")
save_evoked(contrast_b1, "contrast_block-1")
save_evoked(contrast_b2, "contrast_block-2")
save_evoked(contrast_rec, "contrast_rec")


def save_topomap(evoked, name, t=0):
    s = int((t - evoked.tmin) * evoked.info["sfreq"])
    topo = evoked.data[:, s]
    plot_topomap(topo, evoked.info, show=False)
    plt.savefig(
        f"figures/interpretability/erp_{name}_topo.png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        dpi=1000,
    )


save_topomap(contrast_grand_avg, "contrast_grand-avg", t=0.45)
save_topomap(contrast_b1, "contrast_block-1", t=0.30)
save_topomap(contrast_b2, "contrast_block-2", t=0.30)
save_topomap(contrast_rec, "contrast_rec", t=0.30)
