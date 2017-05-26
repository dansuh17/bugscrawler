import pandas as pd

df = pd.read_csv('album_info.csv', index_col=False)

print(df.columns.values)
print(df['앨범명'].as_matrix())
print(df['index'].as_matrix())

