#!/usr/bin/env python

import pdb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df_select = pd.read_csv("data/block_erp_select.csv")
df_select = pd.concat(
    [df_select, pd.read_csv("data/block_mi_select.csv")], ignore_index=True
)
df_select = pd.concat(
    [df_select, pd.read_csv("data/block_lr_select.csv")], ignore_index=True
)
df_select = df_select.reset_index()

idx = ["dataset", "subject", "session", "fold"]
df_score_train = df_select.groupby(idx)["train_score"].aggregate("first")
df_score_val = df_select.groupby(idx)["val_score"].aggregate("first")
df_score_test = df_select.groupby(idx)["test_score"].aggregate("first")
df_diff = df_select.set_index(idx + ["block"])
df_diff["validation"] = df_diff["val_score"] - df_score_val
df_diff["training"] = df_diff["train_score"] - df_score_train
df_diff["test"] = df_diff["test_score"] - df_score_test
df_diff = df_diff.reset_index()
df_diff = df_diff.melt(
    id_vars=idx + ["block", "n_features"],
    value_vars=["training", "validation", "test"],
    var_name="split",
    value_name="roc_auc_diff",
)

df_diff["n_blocks"] = df_diff["block"] + 1
df_diff["roc_auc_diff_%"] = df_diff["roc_auc_diff"] * 100
fig, axs = plt.subplots(2, layout="tight", sharex=True)

df_diff = df_diff[df_diff['split']=='test']
datasets = [["BNCI2014-008", "BNCI2015-003"], ["BNCI2014-001", "BNCI2014-004"]]
for i, selection in enumerate(datasets):
    axs[i].set_ylabel("NMSE")
    sns.scatterplot(
        data=df_diff[df_diff["dataset"].isin(selection)],
        x="n_features",
        hue="dataset",
        y="roc_auc_diff_%",
        style="split",
        ax=axs[i],
        legend=True,
    )
    axs[i].set_xscale('log')
    axs[i].axhline(0, color='black',linestyle='--') 
plt.show()
