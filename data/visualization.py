import matplotlib.pyplot as plt
import random
from PIL import Image
from matplotlib.patches import Rectangle

from data_loader import DataLoader

# visualizer based off https://www.kaggle.com/eduardo4jesus/stanford-cars-dataset-a-quick-look-up

def get_assets(df, i):
    image = Image.open(df['fname'][i])
    title = df['labels'][i] if 'labels' in df else 'Unclassified'

    xy = df['bbox_x1'][i], df['bbox_y1'][i]
    width = df['bbox_x2'][i] - df['bbox_x1'][i]
    height = df['bbox_y2'][i] - df['bbox_y1'][i]
    rect = Rectangle(xy, width, height, fill=False, color='r', linewidth=2)
    
    return (image, title, rect)

def display_image(df, i):
    image, title, rect = get_assets(df, i)    
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.gca().add_patch(rect)

def display_images(n):

    fig, ax = plt.subplots(n, 2, figsize=(15, 5*n))
    train_len = len(data.df_train)
    test_len = len(data.df_test)

    for i in range(n):
        
        im, title, rect = get_assets(data.df_train, random.randint(0, train_len))
        sub = ax[i, 0]
        sub.imshow(im)
        sub.axis('off')
        sub.set_title(title)
        sub.add_patch(rect)
        
        im, title, rect = get_assets(data.df_test, random.randint(0, test_len))
        sub = ax[i, 1]
        sub.imshow(im)
        sub.axis('off')
        sub.set_title(title)
        sub.add_patch(rect)
        
    plt.show()

data = DataLoader()
display_image(data.df_train, 0)
display_images(5)

