import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from scipy.io import loadmat
from tensorflow.data.experimental import sample_from_datasets, AUTOTUNE

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
pd.set_option('display.width', 1000)

IMG_SIZE = 224
BATCH_SIZE = 32
BUFFER_SIZE = 100000

# loader based off 
# https://www.kaggle.com/eduardo4jesus/stanford-cars-dataset-a-quick-look-up

class DataLoader():
    
    def __init__(self):
        devkit_path = Path('devkit')
        
        meta = loadmat(devkit_path/'cars_meta.mat')
        train_annos = loadmat(devkit_path/'cars_train_annos.mat')
        test_annos = loadmat(devkit_path/'cars_test_annos.mat')
        
        labels = [c for c in meta['class_names'][0]]
        labels = pd.DataFrame(labels, columns=['labels'])
        
        frame = [[i.flat[0] for i in line] 
        for line in train_annos['annotations'][0]]
        columns_train = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 
                         'label', 'fname']
        df_train = pd.DataFrame(frame, columns=columns_train)
        df_train['label'] = df_train['label']-1 # indexing starts on zero.
        df_train['fname'] = [f'cars_train/{f}' 
                for f in df_train['fname']] #  Appending Path
        df_train = df_train.sort_index()
        
        frame = [[i.flat[0] for i in line] 
        for line in test_annos['annotations'][0]]
        columns_test = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
        df_test = pd.DataFrame(frame, columns=columns_test)
        df_test['fname'] = [f'cars_test/{f}' 
               for f in df_test['fname']] #  Appending Path

        self.df_train = df_train
        self.df_test = df_test
        self.labels = labels
        
    def get_pipeline(self, apply_aug=True, seed=None, test=False):
        ds = None
        datasets = []
        df = self.df_train if not test else self.df_test
       
        for car_type in df['label'].unique():
            cars = df[df['label']==car_type]
            paths = cars['fname']
            labels = cars['label']
            paths_labels_ds = tf.data.Dataset.from_tensor_slices((paths, labels)).cache()
            paths_labels_ds = paths_labels_ds.shuffle(BUFFER_SIZE).repeat()
            img_labels_ds = paths_labels_ds.map(load_and_resize_image, num_parallel_calls=AUTOTUNE)
            if apply_aug:
                img_labels_ds = img_labels_ds.map(augment_img)
            img_labels_ds = img_labels_ds.map(standard_scaler)
            datasets.append(img_labels_ds)
            
        num_labels = len(self.labels)
        sampling_weights = np.ones(num_labels)*(1./num_labels)

        ds = sample_from_datasets(datasets, weights=sampling_weights, seed=seed)
        ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
        
        return ds    

# =============================================================================
# HELPER PREPROCESS FUNCTIONS
# =============================================================================

def standard_scaler(img, label):
    img = img/255
    return img, label

def load_and_resize_image(path, label, channels=1):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

def augment_img(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, .1)
    img = tf.image.random_jpeg_quality(img, 50, 100)
    return img, label



