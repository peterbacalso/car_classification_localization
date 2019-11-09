import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from data_loader import DataLoader

def plot_dist(data, y_label, x_label):
    data.head(50).plot.bar(figsize=(15,10))
    plt.xticks(rotation=90);
    plt.xlabel(x_label);
    plt.ylabel(y_label);    
    plt.show()
    

if __name__=="__main__":
    data = DataLoader(batch_size=32)
    
    df_train = data.df_train.merge(data.labels, 
                                   left_on='label', 
                                   right_index=True)
    df_train = df_train.sort_index()
    freq_labels = df_train.groupby('labels').count()[['label']]
    freq_labels = freq_labels.rename(columns={'label': 'count'})
    freq_labels = freq_labels.sort_values(by='count', ascending=False)
    
    ds = data.get_pipeline(onehot=False, apply_aug=False, channels=3)
    labels = []
    for inputs, outputs in ds.take(300):
        labels.extend(outputs['labels'].numpy())
# =============================================================================
#         print("=============INPUTS================================")
#         print(inputs) # image matrix
#         print()
#         print("=============OUTPUTS================================")
#         print(outputs['labels']) # (labels, bbox)
# =============================================================================
    labels = pd.Series(labels)
    label_dict = data.labels.to_dict()['labels']
    df = labels.map(label_dict)
    oversample_freq_labels = df.value_counts()
    
    # PLOT UNDERLYING CLASS DISTRIBUTIONS
    plot_dist(freq_labels, "Count", "Cars (with class imbalance)")
    
    # PLOT CLASS DISTRIBUTION AFTER OVERSAMPLING
    plot_dist(oversample_freq_labels, "Count", 
              "Cars (balanced with oversampling)")


