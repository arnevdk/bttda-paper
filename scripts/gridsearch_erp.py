import pandas as pd

metrics = ['test_score', 'test_mse']

df = pd.read_csv('data/gridsearch_erp_raw.csv')
df = df.groupby(['n_blocks', 'theta'])[metrics].aggregate('mean').reset_index()
df.to_csv('data/gridsearch_erp_processed.csv')
