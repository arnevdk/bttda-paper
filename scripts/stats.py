import pdb

import pandas as pd
import scipy.stats

results = pd.read_csv("data/results_erp.csv")
results = results.groupby(["dataset", "session", "subject", "pipeline"])
results = results["score"].aggregate("mean")

datasets = ["BNCI2014-008", "BNCI2015-003"]
pairs = [("BTTDA_10", "HODA"), ("PARAFACDA_10", "HODA"), ("BTTDA_10", "PARAFACDA_10")]

for dataset in datasets:
    print(dataset)
    df = results[results.index.get_level_values(0) == dataset]
    for decoder1, decoder2 in pairs:
        score1 = df[df.index.get_level_values(-1) == decoder1]
        score2 = df[df.index.get_level_values(-1) == decoder2]
        stat, p = scipy.stats.wilcoxon(score1, score2, alternative="greater")
        print(f"{decoder1} > {decoder2}: p={p}")
    print()

results = pd.read_csv("data/results_lr.csv")
results = results.groupby(["dataset", "session", "subject", "pipeline"])
results = results["score"].aggregate("mean")
print("=======================================================================")
datasets = ["BNCI2014-004"]
pairs = [("BTTDA_10", "HODA"), ("PARAFACDA_10", "HODA"), ("BTTDA_10", "PARAFACDA_10")]

for dataset in datasets:
    print(dataset)
    df = results[results.index.get_level_values(0) == dataset]
    for decoder1, decoder2 in pairs:
        score1 = df[df.index.get_level_values(-1) == decoder1]
        score2 = df[df.index.get_level_values(-1) == decoder2]
        stat, p = scipy.stats.wilcoxon(score1, score2, alternative="greater")
        print(f"{decoder1} > {decoder2}: p={p}")
    print()
