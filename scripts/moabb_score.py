import math
import pdb

import numpy as np
import pandas as pd


def wrapped_table(df, wrap=5):
    """Construct wrapped table."""
    val_cols = df.columns[1:]
    table = ""
    n_rows = int(math.ceil(len(val_cols) / wrap))
    for r in range(n_rows):
        row_cols = ["Pipelines"] + val_cols[r * wrap : (r + 1) * wrap].tolist()
        print(row_cols)
        df_row = df[row_cols]
        table_rows = df_row.to_latex(
            formatters={
                c: (
                    lambda v: ("\\boldmath" if v[2] else "")
                    + f"${v[0]:.2f}\pm{v[1]:.2f}$"
                )
                for c in val_cols
            },
            escape=False,
            column_format="@{}l" + "r" * len(val_cols) + "@{}",
            index=False,
            na_rep="-",
        )

        table_rows_moabb = "\n".join(table_rows.split("\n")[: 6 + 3])
        table_rows_own = "\n".join(table_rows.split("\n")[6 + 3 :]) + "\n"
        # table_rows = table_rows_moabb + "\\midrule \n" + table_rows_own
        table_rows = table_rows_moabb + table_rows_own

        if r < n_rows - 1:
            table_rows = "\n".join(table_rows.split("\n")[:-4]) + "\n"
        if r > 0:
            table_rows = "\\midrule \n" + "\n".join(table_rows.split("\n")[2:]) + "\n"
        table += table_rows
    return table


def indicate_max(df):
    val_cols = df.columns[1:]
    df_max = df[val_cols].map(lambda x: x[0], na_action="ignore")
    df_max = df_max.aggregate("max")
    for c in val_cols:
        df[c] = df[c].apply(
            lambda x: np.nan if np.any(np.isnan(x)) else (*x, x[0] == df_max[c])
        )
    return df


def format_moabb_results(df):
    val_cols = df.columns[1:]
    split = df[val_cols].map(lambda x: [float(v) for v in x.split("±")])
    df[val_cols] = split
    return df


def format_own_results(df_res):
    df_res = df_res.replace(
        {
            "BTTDA": "BTTDA+LDA",
            "HODA": "HODA+LDA",
            "PARAFACDA": "PARAFACDA+LDA",
        }
    )
    df_res = df_res.groupby(["dataset", "pipeline"])
    df_res = df_res.score.aggregate(["mean", "std"])
    df_res *= 100
    df_res["val"] = list(zip(df_res["mean"], df_res["std"]))
    df_res = df_res.reset_index().pivot(
        index="pipeline", columns="dataset", values="val"
    )
    df_res = df_res.reset_index()
    df_res = df_res.rename(columns=dict(pipeline="Pipelines"))
    return df_res


def calculate_average(df):
    """Calculate average column"""
    val_cols = df.columns[1:]
    mean_score = df[val_cols].map(lambda x: x[0], na_action="ignore")
    mean_score = mean_score.aggregate(np.nanmean, axis=1)
    mean_std = df[val_cols].map(lambda x: x[1], na_action="ignore")
    mean_std = mean_std.aggregate(np.nanstd, axis=1)
    df["Average"] = list(zip(mean_score, mean_std))
    print(df)
    return df


def sort_datasets(df):
    """Sort dataset names."""
    val_cols = df.columns[1:]
    df_sorted = pd.DataFrame()
    df_sorted["Pipelines"] = df["Pipelines"]
    df_sorted[sorted(val_cols)] = df[sorted(val_cols)]
    df = df_sorted
    avg = df.pop("Average")
    df.insert(len(df.columns), "Average", avg)
    return df


def sort_pipelines(df):
    """Order pipelines"""
    order = list(range(len(df)))
    order = order[:-3] + [-2, -1, -3]
    df = df.iloc[order, :]
    return df


def generate_table(df):
    df = sort_datasets(df)
    df = sort_pipelines(df)
    df = indicate_max(df)
    table = wrapped_table(df)
    return table


# ERPs ========================================================================
# Read and format own results
df_res = pd.read_csv("data/results_erp.csv")
df_res = format_own_results(df_res)


# Read and format MOABB ERP results
df_moabb = pd.read_csv("data/moabb_erp.csv")
df_moabb = format_moabb_results(df_moabb)
keep_cols = [c for c in df_moabb.columns if c in df_res.columns]
df_moabb = df_moabb[keep_cols]
df = pd.concat([df_moabb, df_res], ignore_index=True)


def agg_tuples(tuples):
    mean = np.mean([t[0] for t in tuples if isinstance(t, tuple)])
    std = np.sqrt(
        np.nanmean(
            [t[1] ** 2 for t in tuples if isinstance(t, tuple) or isinstance(t, list)]
        )
    )
    return (mean, std)


df = df.set_index("Pipelines")
df = df.map(tuple)
df.loc[:, "Average"] = df.aggregate(agg_tuples, axis=1)
df = df.reset_index()

# Generate and save table
table = generate_table(df)
path = "tables/score_erp.tex"
with open(path, "w") as file:
    file.write(table)

# MI ========================================================================
# Read and format own results
df_res = pd.read_csv("data/results_mi.csv")
df_res = format_own_results(df_res)

# Read and format MOABB ERP results
df_moabb = pd.read_csv("data/moabb_mi.csv")
df_moabb = format_moabb_results(df_moabb)
keep_cols = [c for c in df_moabb.columns if c in df_res.columns]
df_moabb = df_moabb[keep_cols]
keep_rows = ["ACM+TS+SVM", "TS+EL", "FgMDM", "ShallowConvNet", "EEGTCNet"]
df_moabb = df_moabb[df_moabb["Pipelines"].isin(keep_rows)]

df = pd.concat([df_moabb, df_res], ignore_index=True)


df = df.set_index("Pipelines")
df = df.map(tuple)
df.loc[:, "Average"] = df.aggregate(agg_tuples, axis=1)
df = df.reset_index()

# Generate and save table
table = generate_table(df)
path = "tables/score_mi.tex"
with open(path, "w") as file:
    file.write(table)
