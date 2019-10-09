import pandas as pd
from pathlib import Path
from scipy.io import loadmat

# loader based off https://www.kaggle.com/eduardo4jesus/stanford-cars-dataset-a-quick-look-up

class DataLoader():
    
    def __init__(self):
        train_path = Path('cars_train')
        test_path = Path('cars_test')
        devkit_path = Path('devkit')
        
        meta = loadmat(devkit_path/'cars_meta.mat')
        train_annos = loadmat(devkit_path/'cars_train_annos.mat')
        test_annos = loadmat(devkit_path/'cars_test_annos.mat')
        
        labels = [c for c in meta['class_names'][0]]
        labels = pd.DataFrame(labels, columns=['labels'])
        
        frame = [[i.flat[0] for i in line] for line in train_annos['annotations'][0]]
        columns_train = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
        df_train = pd.DataFrame(frame, columns=columns_train)
        df_train['class'] = df_train['class']-1 # Python indexing starts on zero.
        df_train['fname'] = [train_path/f for f in df_train['fname']] #  Appending Path
        df_train = df_train.merge(labels, left_on='class', right_index=True)
        df_train = df_train.sort_index()
        
        frame = [[i.flat[0] for i in line] for line in test_annos['annotations'][0]]
        columns_test = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
        df_test = pd.DataFrame(frame, columns=columns_test)
        df_test['fname'] = [test_path/f for f in df_test['fname']] #  Appending Path

        self.df_train = df_train
        self.df_test = df_test

