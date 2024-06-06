#!/usr/bin/env python

import pdb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from setup import set_size, textwidth

df = pd.read_csv("data/erp_blocks.csv")

fig, axs = plt.subplots(2, 3, layout="tight", sharex=True)
lineplot_kwargs = dict(
    data=df,
    x="block",
    hue="dataset",
    # style="session",
    errorbar=None,
    # units="subject",
    # estimator=None,
)
sns.lineplot(y="F_rt", ax=axs[0, 0], legend=True, **lineplot_kwargs)
axs[0, 0].set_ylabel("$\phi$")
axs[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
sns.lineplot(y="mse", ax=axs[0, 1], legend=False, **lineplot_kwargs)
axs[0, 1].set_ylabel("MSE")
sns.lineplot(y="val_score", ax=axs[0, 2], legend=False, **lineplot_kwargs)
axs[0, 2].set_ylabel("ROC-AUC")
for ax in axs.flatten():
    ax.set_xlabel("$b$")
fig.set_size_inches(set_size(width_pt=textwidth, subplots=(2, 3)))
fig.tight_layout()
fig.savefig("figures/blocks.pgf", bbox_inches="tight", pad_inches=0)
fig.savefig("figures/blocks.svg", bbox_inches="tight", pad_inches=0)
fig.savefig("figures/blocks.pdf", bbox_inches="tight", pad_inches=0)
