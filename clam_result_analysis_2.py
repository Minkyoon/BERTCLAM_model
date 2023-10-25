import pandas as pd

df = pd.read_csv('/home/jsy/2023_clam/results/remission_multimodal_stratified_721_s1/summary.csv')

print('auc: ', df['test_auc'].mean())
print('acc: ', df['test_acc'].mean())