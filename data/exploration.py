import matplotlib.pyplot as plt

from data_loader import DataLoader

data = DataLoader()

df_train = data.df_train.merge(data.labels, left_on='label', right_index=True)
df_train = df_train.sort_index()
        
freq_labels = df_train.groupby('labels').count()[['label']]
freq_labels = freq_labels.rename(columns={'label': 'count'})
freq_labels = freq_labels.sort_values(by='count', ascending=False)
print(freq_labels.head())

freq_labels.head(50).plot.bar(figsize=(15,10))
plt.xticks(rotation=90);
plt.xlabel("Cars");
plt.ylabel("Count");
