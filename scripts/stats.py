import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from moabb.analysis.meta_analysis import (combine_effects, combine_pvalues,
                                          compute_dataset_statistics,
                                          find_significant_differences)
from moabb.analysis.plotting import meta_analysis_plot, summary_plot

df = pd.read_csv("data/results_mi.csv")
df_stats = compute_dataset_statistics(df)
df_sig, df_effect = find_significant_differences(df_stats)


def p_to_star(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


for pair in [("BTTDA", "HODA"), ("BTTDA", "PARAFACDA"), ("PARAFACDA", "HODA")]:
    df_stats_pair = df_stats.query(f"pipe1=='{pair[0]}' and pipe2=='{pair[1]}'")
    avg_nsub = df_stats_pair["nsub"].mean()
    n_datasets = len(df_stats_pair)
    df_stats_pair.loc[-1] = {
        "dataset": "\\textbf{Meta-effect}",
        "index": len(df_stats_pair),
        "pipe1": pair[0],
        "pipe2": pair[1],
        "p": combine_pvalues(df_stats_pair["p"], avg_nsub),
        "smd": combine_effects(df_stats_pair["smd"], avg_nsub) / n_datasets,
        "nsub": avg_nsub,
    }
    df_stats_pair["p_star"] = df_stats_pair["p"].apply(p_to_star)
    df_stats_pair["dof"] = df_stats_pair["nsub"] - 1
    df_stats_pair["ci"] = scipy.stats.t.ppf(0.95, df_stats_pair["dof"]) / np.sqrt(
        df_stats_pair["nsub"]
    )
    df_stats_pair["dataset_idx"] = np.arange(len(df_stats_pair))

    df_stats_pair.to_csv(f"data/pairwise_stats/mi_{pair[0]}_{pair[1]}.csv")
print(df_sig)
print(df_effect)

# meta_analysis_plot(df_stats, "HODA", "BTTDA")
# plt.show()
# meta_analysis_plot(df_stats, "HODA", "PARAFACDA")
# plt.show()
# meta_analysis_plot(df_stats, "PARAFACDA", "BTTDA")
# plt.show()
# summary_plot(df_sig, df_effect, simplify=False)
# plt.show()
