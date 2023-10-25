import pandas as pd
import numpy as np

train_label = pd.read_csv('/home/jsy/2023_retro_time_adaptive/data/model_results/test.py/tensor_id/train/train_label.csv')
valid_label = pd.read_csv('/home/jsy/2023_retro_time_adaptive/data/model_results/test.py/tensor_id/valid/valid_label.csv')
test_label = pd.read_csv('/home/jsy/2023_retro_time_adaptive/data/model_results/test.py/tensor_id/test/test_label.csv')

import pandas as pd

def expand_dataframe(df, n):
    expanded_data = []
    for _, row in df.iterrows():
        for i in range(n):
            new_row = row.copy()
            new_row['case_id'] = f'{row["case_id"]}_{i}'
            expanded_data.append(new_row)
    return pd.DataFrame(expanded_data)


train_label = expand_dataframe(train_label, 10)
valid_label = expand_dataframe(valid_label, 10)
test_label = expand_dataframe(test_label, 10)


train_label.drop_duplicates(subset=['case_id'], inplace=True)
valid_label.drop_duplicates(subset=['case_id'], inplace=True)
test_label.drop_duplicates(subset=['case_id'], inplace=True)

train_label.reset_index(drop=True, inplace=True)
valid_label.reset_index(drop=True, inplace=True)
test_label.reset_index(drop=True, inplace=True) 

splits_df = pd.DataFrame({
    'train': train_label['case_id'],
    'val': valid_label['case_id'],
    'test': test_label['case_id']
}, dtype=str)

# splits_df = splits_df.applymap(lambda x: 'id_' + str(x) if pd.notna(x) else x)






splits_df.to_csv('/home/jsy/2023_clam/CLAM/splits/crohn_laboratory/splits_0.csv', index=False)

all_ids = train_label['case_id'].tolist() + valid_label['case_id'].tolist() + test_label['case_id'].tolist()




train_bool = [id in train_label['case_id'].tolist() for id in all_ids]
valid_bool = [id in valid_label['case_id'].tolist() for id in all_ids]
test_bool = [id in test_label['case_id'].tolist() for id in all_ids]

all_ids = ['id_' + str(x) if x is not None else x for x in all_ids]

splits_bool_df = pd.DataFrame({
    'train': train_bool,
    'val': valid_bool,
    'test': test_bool
}, index=all_ids)

splits_bool_df.to_csv('/home/jsy/2023_clam/CLAM/splits/crohn_laboratory/splits_0_bool.csv')

train_counts = train_label['label'].value_counts()
valid_counts = valid_label['label'].value_counts()
test_counts = test_label['label'].value_counts()

data = {
    'train': train_counts,
    'val': valid_counts,
    'test': test_counts
}
descriptor_df = pd.DataFrame(data)

descriptor_df.to_csv('splits_0_descriptor.csv')