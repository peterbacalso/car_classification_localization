import matplotlib.pyplot as plt

from data_loader import DataLoader

# exploration based off https://www.kaggle.com/eduardo4jesus/stanford-cars-dataset-a-quick-look-up

data = DataLoader()

freq_labels = data.df_train.groupby('labels').count()[['class']]
freq_labels = freq_labels.rename(columns={'class': 'count'})
freq_labels = freq_labels.sort_values(by='count', ascending=False)
print(freq_labels.head())

freq_labels.head(50).plot.bar(figsize=(15,10))
plt.xticks(rotation=90);
plt.xlabel("Cars");
plt.ylabel("Count");