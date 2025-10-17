import pandas as pd
from moabb.analysis.meta_analysis import compute_dataset_statistics

df = pd.read_csv("data/results_erp.csv")
df_stats = compute_dataset_statistics(df)

df = []
for pair in [("BTTDA", "HODA"), ("BTTDA", "PARAFACDA"), ("PARAFACDA", "HODA")]:
    df_stats_pair = df_stats.query(f"pipe1=='{pair[0]}' and pipe2=='{pair[1]}'")
    df.append(df_stats_pair[["dataset", "pipe1", "pipe2", "p", "smd"]])
df = pd.concat(df, ignore_index=True)

# Prepare format
df["p"] = df["p"].apply("\\num{{{:.2e}}}".format)
df["smd"] = df["smd"].apply("{:.2f}".format)
df = df.rename(
    columns={"pipe1": "decoder 1", "pipe2": "decoder 2", "p": "$p$", "smd": "SMD"}
)

df = df.pivot(
    columns=["decoder 1", "decoder 2"], index="dataset", values=["$p$", "SMD"]
)
df.columns = df.columns.swaplevel(0, 1)
df.columns = df.columns.swaplevel(1, 2)
df = df.sort_index(axis=1)

# Write latex table
table = df.to_latex(
    index=True,
    sparsify=True,
    index_names=False,
    column_format="@{}lrrrrrr@{}",
    multicolumn_format="c",
    multirow=True,
)
path = "tables/stats_mi.tex"
with open(path, "w") as file:
    file.write(table)
print(table)
