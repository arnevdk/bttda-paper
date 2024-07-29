import math
import pdb

import numpy as np
import pandas as pd

# Read and format MOABB results
df = pd.read_csv("data/moabb_erp.csv")
df = df.rename(
    columns={
        "BNCI2014_008": "BNCI2014-008",
        "BNCI2014_009": "BNCI2014-009",
        "BNCI2015_003": "BNCI2015-003",
        "BI2012": "BrainInvaders2012",
        "BI2013a": "BrainInvaders2013a",
        "BI2014a": "BrainInvaders2014a",
        "BI2014b": "BrainInvaders2014b",
        "BI2015a": "BrainInvaders2015a",
        "BI2015b": "BrainInvaders2015b",
        "Cattan2019_VR": "Cattan2019-VR",
        "EPFLP300": "EPFLP300",
        "Huebner2017": "Huebner2017",
        "Huebner201r8": "Huebner2018",
        "Lee2019_ERP": "Lee2019-ERP",
        "Sosulski2019": "Sosulski2019",
    }
)
val_cols = df.columns[1:]
df["DemonsP300"] = np.nan
df[val_cols] = df[val_cols].apply(
    lambda x: [[float(v) for v in r] for r in x.str.split("±")]
)

# Read and format own results
df_res = pd.read_csv("data/results_erp.csv")
df_res = df_res.replace(
    {
        "BTTDA_10": "BTTDA+LDA",
        "HODA": "HODA+LDA",
        "PARAFACDA_10": "PARAFACDA+LDA",
    }
)
df_res = df_res.groupby(["dataset", "pipeline"])
df_res = df_res.score.aggregate(["mean", "std"])
df_res *= 100
df_res["val"] = list(zip(df_res["mean"], df_res["std"]))
df_res = df_res.reset_index().pivot(index="pipeline", columns="dataset", values="val")
df_res = df_res.reset_index()
df_res = df_res.rename(columns=dict(pipeline="Pipelines"))
df = df.append(df_res, ignore_index=True)

# Sort dataset names
val_cols = df.columns[1:]
df_sorted = pd.DataFrame()
df_sorted["Pipelines"] = df["Pipelines"]
df_sorted[sorted(val_cols)] = df[sorted(val_cols)]
df = df_sorted

# Sort dataset rows
order = list(range(len(df)))
order = order[:-3] + [-2, -1, -3]
df = df.iloc[order, :]

# Indicate max
df_max = df[val_cols].applymap(lambda x: x[0], na_action="ignore")
df_max = df_max.aggregate("max")
for c in val_cols:
    df[c] = df[c].apply(
        lambda x: np.nan if np.any(np.isnan(x)) else (*x, x[0] == df_max[c])
    )

# Drop Sosulski dataset
del df["Sosulski2019"]
val_cols = df.columns[1:]

# Construct wrapped table
wrap = 5
table = ""
n_rows = int(math.ceil(len(val_cols) / wrap))
for r in range(n_rows):
    row_cols = ["Pipelines"] + val_cols[r * wrap : (r + 1) * wrap].tolist()
    print(row_cols)
    df_row = df[row_cols]
    table_rows = df_row.to_latex(
        formatters={
            c: (lambda v: ("\\boldmath" if v[2] else "") + f"${v[0]:.2f}\pm{v[1]:.2f}$")
            for c in val_cols
        },
        escape=False,
        column_format="@{}l" + "c" * len(val_cols) + "@{}",
        index=False,
        na_rep="-",
    )
    if r < n_rows - 1:
        table_rows = "\n".join(table_rows.split("\n")[:-3]) + "\n"
    if r > 0:
        table_rows = "\\midrule\n" + "\n".join(table_rows.split("\n")[2:]) + "\n"
    table += table_rows

path = "include/moabb_erp_score.tex"
with open(path, "w") as file:
    file.write(table)
