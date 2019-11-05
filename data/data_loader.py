import pandas as pd
import numpy as np
import tensorflow as tf

from functools import partial
from pathlib import Path
from scipy.io import loadmat
from tensorflow.data.experimental import sample_from_datasets, AUTOTUNE
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

IMG_SIZE = 224
BUFFER_SIZE = 100000

class DataLoader():
    
    def __init__(self, train_path='cars_train', 
                 test_path='cars_test', devkit='devkit', 
                 batch_size=32, valid_split=.2):
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
        #df = df[df['label']<=75] # start with small sample for tuning initial hyperparams
        #df = df[(df['label']>3) & (df['label']<=5)]
        
        df_train, df_valid = train_test_split(df, test_size=valid_split)
        
        df_train = df_train.sort_index()
        df_valid = df_valid.sort_index()
        
        test_frame = [[i.flat[0] for i in line] 
                for line in test_annos['annotations'][0]]
        
        columns_test = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
        df_test = pd.DataFrame(test_frame, columns=columns_test)
        df_test['fname'] = [f'{test_path}/{f}' 
               for f in df_test['fname']] #  Appending Path
        
        augmenter = iaa.Sequential([
            #iaa.Resize({"height": IMG_SIZE, "width": IMG_SIZE}),
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.labels = labels
        self.batch_size = batch_size
        self.augmenter = augmenter
        
    def get_pipeline(self, type='train', output='label_bbox', channels=1,
                     apply_aug=True, onehot=True, seed=None):
        '''
        Input:
            type:           Can be 'train', 'validation', or 'test'
            output:         Determines the output values of the pipline. 
                            Can be one of 'label', 'bbox', or 'label_bbox'.
            channels:       Number of channels of the output image (1-3)
            apply_aug:      Bool that determines whether to apply augmentation
            onehot:         Bool that determines whether to 
                            one hot encode the class labels
            seed:           Random seed number
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
            
        if type == 'test':
            paths = df['fname']
            paths_targets = tf.data.Dataset.from_tensor_slices(paths).cache()
            paths_targets = paths_targets.shuffle(BUFFER_SIZE)
            img_targets = paths_targets.map(
                    partial(load_and_resize_image_test, channels=channels),
                    num_parallel_calls=AUTOTUNE)
            img_targets = img_targets.map(standard_scaler_test).repeat()
            ds = img_targets.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
            return ds
        else:
            one_hot_labels = pd.get_dummies(df['label'], prefix=['label'])
            if type=='validation':
                # get the columns in train that are not in valid
                one_hot_train_labels = pd.get_dummies(self.df_train['label'], prefix=['label'])
                col_to_add = np.setdiff1d(one_hot_train_labels.columns, one_hot_labels.columns)
                for c in col_to_add:
                    one_hot_labels[c] = 0
                # select and reorder the validation columns using the train columns
                one_hot_labels = one_hot_labels[one_hot_train_labels.columns]

            for car_type in df['label'].unique():
                cars = df[df['label']==car_type]
                paths = cars['fname']
                labels = one_hot_labels[df['label']==car_type] if onehot else cars['label']
                #bbox = np.log(cars[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']])
                bbox = cars[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]
                paths = tf.data.Dataset.from_tensor_slices(paths)
                if output=='label_bbox':
                    targets = tf.data.Dataset.from_tensor_slices((
                        tf.cast(labels.values, tf.uint8), 
                        tf.cast(bbox.values, tf.int16)
                    ))
                elif output == 'label':
                    targets = tf.data.Dataset.from_tensor_slices(
                            tf.cast(labels.values, tf.uint8))
                elif output == 'bbox':
                    targets = tf.data.Dataset.from_tensor_slices(
                        tf.cast(bbox.values, tf.int16))
                paths_targets = tf.data.Dataset.zip((paths, targets)).cache()
                paths_targets = paths_targets.shuffle(BUFFER_SIZE)
                img_targets = paths_targets.map(
                        partial(load_image, channels=channels),
                        num_parallel_calls=AUTOTUNE)
                if apply_aug:
                    img_targets = img_targets.map(
                            partial(augment_img, 
                                    augmenter=self.augmenter,
                                    output_type=output))
# =============================================================================
#                 img_targets = paths_targets.map(
#                         resize_image,
#                         num_parallel_calls=AUTOTUNE)
# =============================================================================
                #img_targets = img_targets.map(standard_scaler).repeat()
                img_targets = img_targets.repeat()
                datasets.append(img_targets)
            
        num_labels = len(df['label'].unique())
        sampling_weights = np.ones(num_labels)*(1./num_labels)

        ds = sample_from_datasets(datasets, 
                                  weights=sampling_weights, seed=seed)
        ds = ds.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        
        return ds    

# =============================================================================
# HELPER PREPROCESS FUNCTIONS
# =============================================================================

def standard_scaler(img, outputs):
    img = img/255.0
    return img, outputs

def load_image(path, outputs, channels=1):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.cast(img, tf.uint8)
    return img, outputs

def augment_img(img, outputs, augmenter, output_type):
    bbox = None
    if output_type == 'label':
        bbox = outputs
    elif output_type == 'label_bbox':
        bbox = outputs[1]
    
    def aug_mapper(img, bbox):
        bb_prior = BoundingBoxesOnImage([
            BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        ], shape=img.shape)
        before = bb_prior.bounding_boxes[0]
        print('bbox before', before.x1, before.y1, before.x2, before.y2)
        img_aug, bb_aug = augmenter(images=[img], bounding_boxes=bb_prior)
        after = bb_aug.bounding_boxes[0]
        print('bbox after', after.x1, after.y1, after.x2, after.y2)
        return augmenter(images=[img])
    
    #img = augmenter(images=img)
    #img = tf.map_fn(augmenter)
    img_dtype = img.dtype
    img = tf.numpy_function(aug_mapper, [img, bbox], img_dtype)
    
# =============================================================================
#     after = bb_aug.bounding_boxes[0]
#     print('bbox after', after.x1, after.y1, after.x2, after.y2)
# =============================================================================
        
    return img, outputs



# =============================================================================
# HELPERS FOR TEST
# =============================================================================

def standard_scaler_test(img):
    img = img/255
    return img

def load_and_resize_image_test(path, channels=1):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img



