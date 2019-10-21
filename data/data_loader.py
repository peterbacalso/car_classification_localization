import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from scipy.io import loadmat
from tensorflow.data.experimental import sample_from_datasets, AUTOTUNE
from sklearn.model_selection import train_test_split

# =============================================================================
# pd.options.display.max_rows = 500
# pd.options.display.max_columns = 500
# pd.set_option('display.width', 1000)
# =============================================================================

IMG_SIZE = 224
BUFFER_SIZE = 100000

# loader based off 
# https://www.kaggle.com/eduardo4jesus/stanford-cars-dataset-a-quick-look-up

class DataLoader():
    
    def __init__(self, train_path='cars_train', 
                 test_path='cars_test', devkit='devkit', 
                 batch_size=32, valid_split=.15):
        devkit_path = Path(devkit)
        
        meta = loadmat(devkit_path/'cars_meta.mat')
        train_annos = loadmat(devkit_path/'cars_train_annos.mat')
        test_annos = loadmat(devkit_path/'cars_test_annos.mat')
        
        labels = [c for c in meta['class_names'][0]]
        labels = pd.DataFrame(labels, columns=['labels'])
        
        frame = [[i.flat[0] for i in line] 
                for line in train_annos['annotations'][0]]
        
        columns_train = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 
                         'label', 'fname']
        
        df = pd.DataFrame(frame, columns=columns_train)
        df['label'] = df['label']-1 # indexing starts on zero.
        df['fname'] = [f'{train_path}/{f}' 
                for f in df['fname']] #  Appending Path
        
        df_train, df_valid = train_test_split(df, test_size=valid_split)
        
        df_train = df_train.sort_index()
        df_valid = df_valid.sort_index()
        
        test_frame = [[i.flat[0] for i in line] 
                for line in test_annos['annotations'][0]]
        
        columns_test = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
        df_test = pd.DataFrame(test_frame, columns=columns_test)
        df_test['fname'] = [f'{test_path}/{f}' 
               for f in df_test['fname']] #  Appending Path

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.labels = labels
        self.batch_size = batch_size
        
    def get_pipeline(self, type='train', apply_aug=True, seed=None):
        '''
        Input:
            type: can be 'train', 'validation', or 'test'
            apply_aug: bool that determines whether to apply augmentation
            seed: random seed number
        Output:
            image generator
        '''
        ds = None
        datasets = []
        df = []
        if type == 'train':
            df = self.df_train
        elif type == 'validation':
            df = self.df_valid
        elif type == 'test':
            df = self.df_test
       
        for car_type in df['label'].unique():
            cars = df[df['label']==car_type]
            paths = cars['fname']
            labels = cars['label']
            bbox = cars[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]
            paths = tf.data.Dataset.from_tensor_slices(paths)
# =============================================================================
#             targets = tf.data.Dataset.zip((labels, bbox)).cache()
#             paths_targets_ds = tf.data.Dataset.from_tensor_slices((
#                     paths, 
#                     tf.cast(targets.values, tf.int32)
#                     ))
# =============================================================================
            targets = tf.data.Dataset.from_tensor_slices((
                tf.cast(labels.values, tf.int32), 
                tf.cast(bbox.values, tf.int32)
            ))
            paths_targets_ds = tf.data.Dataset.zip((paths, targets)).cache()
            paths_targets_ds = paths_targets_ds.shuffle(BUFFER_SIZE)
            img_targets_ds = paths_targets_ds.map(
                    load_and_resize_image, num_parallel_calls=AUTOTUNE)
            if apply_aug:
                img_targets_ds = img_targets_ds.map(augment_img)
            img_targets_ds = img_targets_ds.map(standard_scaler).repeat()
            datasets.append(img_targets_ds)
            
        num_labels = len(self.labels)
        sampling_weights = np.ones(num_labels)*(1./num_labels)

        ds = sample_from_datasets(datasets, 
                                  weights=sampling_weights, seed=seed)
        ds = ds.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        
        return ds    

# =============================================================================
# HELPER PREPROCESS FUNCTIONS
# =============================================================================

def standard_scaler(img, outputs):
    img = img/255
    return img, outputs

def load_and_resize_image(path, outputs, channels=1):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, outputs

def augment_img(img, outputs):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, .1)
    img = tf.image.random_jpeg_quality(img, 50, 100)
    return img, outputs



