#!/usr/bin/env python

import pdb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from setup import set_size, textwidth

df_select = pd.read_csv("data/block_erp_select.csv")
df_select = pd.concat(
    [df_select, pd.read_csv("data/block_mi_select.csv")], ignore_index=True
)
df_select = df_select.reset_index()
df_train = pd.read_csv("data/block_erp_train.csv")
df_train = pd.concat(
    [df_train, pd.read_csv("data/block_mi_train.csv")], ignore_index=True
)
df_train = df_train.reset_index()

df_select = df_select.groupby(["dataset", "subject", "session", "block"])
df_select = df_select[["train_score", "val_score"]].aggregate("mean")
df_select = df_select.reset_index()
idx = ["dataset", "subject", "session"]
df_score_train = df_select.groupby(idx)["train_score"].aggregate("first")
df_score_val = df_select.groupby(idx)["val_score"].aggregate("first")
df_diff = df_select.set_index(idx + ["block"])
df_diff["validation"] = df_diff["val_score"] - df_score_val
df_diff["training"] = df_diff["train_score"] - df_score_train
df_diff = df_diff.reset_index()
df_diff = df_diff.melt(
    id_vars=idx + ["block"],
    value_vars=["training", "validation"],
    var_name="split",
    value_name="roc_auc_diff",
)

df_diff["n_blocks"] = df_diff["block"] + 1
df_diff["roc_auc_diff_%"] = df_diff["roc_auc_diff"] * 100
fig, axs = plt.subplots(1, 2, layout="tight", sharex=True)
lineplot_kwargs = dict(
    x="n_blocks",
    hue="dataset",
    errorbar=None,
)

df_train["n_blocks"] = df_train["block"]

sns.lineplot(data=df_train, y="nmse", ax=axs[0], legend=True, **lineplot_kwargs)
axs[0].set_ylabel("NMSE")
sns.lineplot(
    data=df_diff,
    y="roc_auc_diff_%",
    style="split",
    ax=axs[1],
    legend=True,
    **lineplot_kwargs
)
axs[1].set_ylim([0, None])
axs[1].set_ylabel("$\Delta$ROC-AUC (\%.)")
for ax in axs.flatten():
    ax.set_xlabel("$b$")
    ax.set_xlim([1, 10])
fig.set_size_inches(set_size(width_pt=textwidth, subplots=(1, 2)))
fig.tight_layout()
fig.savefig("figures/blocks.pgf", bbox_inches="tight", pad_inches=0)
fig.savefig("figures/blocks.svg", bbox_inches="tight", pad_inches=0)
fig.savefig("figures/blocks.pdf", bbox_inches="tight", pad_inches=0)
