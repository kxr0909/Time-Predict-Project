"""
将阶段4 Excel数据转换为训练用CSV格式
目标：用 ELEC + 气候特征 填补 HEAT
"""
import pandas as pd
import os
import numpy as np

input_dir = 'real_data/4-1h外连续补充数据'
output_dir = 'real_data/training_data'
os.makedirs(output_dir, exist_ok=True)

feature_cols = ['ELEC', 'GAS', 'TEM', 'REHUM', 'RAINF', 'PRE', 'WINDSPEED', 'U', 'V', 'WINDDIR']
target_col = 'HEAT'

dfs = []
for year in range(2019, 2026):
    fname = f'填补完成_{year}气象数据_已替换变量.xlsx'
    fpath = os.path.join(input_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_excel(fpath)
        cols_needed = ['TIME'] + feature_cols + [target_col]
        df = df[cols_needed].copy()
        df.columns = ['date'] + feature_cols + [target_col]
        dfs.append(df)
        print(f'{year}: {len(df):,} rows')

df_all = pd.concat(dfs, ignore_index=True)
print(f'\nTotal: {len(df_all):,} rows')
print(f'Time range: {df_all["date"].min()} ~ {df_all["date"].max()}')

csv_path = os.path.join(output_dir, 'energy_climate.csv')
df_all.to_csv(csv_path, index=False)
print(f'Saved to: {csv_path}')

print('\n=== Data Stats ===')
for col in feature_cols + [target_col]:
    print(f'{col:12s}: min={df_all[col].min():>12.2f}, max={df_all[col].max():>12.2f}, mean={df_all[col].mean():>12.2f}')

print('\nDone!')
