import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
from matplotlib.patches import Rectangle

from data_loader import DataLoader

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
pd.set_option('display.width', 1000)


def get_assets(df, i):
    image = Image.open(df['fname'].iloc[i])
    title = df['labels'].iloc[i] if 'labels' in df else 'Unclassified'

    xy = df['bbox_x1'].iloc[i], df['bbox_y1'].iloc[i]
    width = df['bbox_x2'].iloc[i] - df['bbox_x1'].iloc[i]
    height = df['bbox_y2'].iloc[i] - df['bbox_y1'].iloc[i]
    rect = Rectangle(xy, width, height, fill=False, color='r', linewidth=2)
    
    return (image, title, rect)

def display_image(df, i):
    image, title, rect = get_assets(df, i)    
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.gca().add_patch(rect)
    
    plt.show()

def display_images(df_train, df_test, n):

    fig, ax = plt.subplots(n, 2, figsize=(15, 5*n))
    train_len = len(df_train)
    test_len = len(df_test)

    for i in range(n):
        
        im, title, rect = get_assets(df_train, random.randint(0, train_len))
        sub = ax[i, 0]
        sub.imshow(im)
        sub.axis('off')
        sub.set_title(title)
        sub.add_patch(rect)
        
        im, title, rect = get_assets(df_test, random.randint(0, test_len))
        sub = ax[i, 1]
        sub.imshow(im)
        sub.axis('off')
        sub.set_title(title)
        sub.add_patch(rect)
        
    plt.show()

data = DataLoader()

df_train = data.df_train.merge(data.labels, left_on='label', right_index=True)
df_train = df_train.sort_index()

# =============================================================================
# for i in range(df_train.shape[0]):
#     display_image(df_train, i)
# =============================================================================
    
display_image(df_train, 0)
display_images(df_train, data.df_test, 20)

