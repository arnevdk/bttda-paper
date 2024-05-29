import pandas as pd

moabb_res_erp = pd.read_csv("data/moabb_erp.csv")
moabb_res_erp.to_latex(buf="include/score_erp.tex", index=False, escape=True)
