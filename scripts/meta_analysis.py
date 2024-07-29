import pdb

import moabb.analysis.plotting as moabb_plt
import pandas as pd
from moabb.analysis.meta_analysis import (compute_dataset_statistics,
                                          find_significant_differences)

results = pd.read_csv("data/moabb_results_erp.csv")
results = results[~results["dataset"].isin(["BrainInvaders2014b", "Cattan2019-VR"])]
stats = compute_dataset_statistics(results)
fig = moabb_plt.meta_analysis_plot(stats, "BTTDA_10", "PARAFACDA_10")
fig.savefig("figures/meta.pgf")
fig.savefig("figures/meta.png")
