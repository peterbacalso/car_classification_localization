import pandas as pd
import numpy as np
import tensorflow as tf

from functools import partial
from pathlib import Path
from scipy.io import loadmat
from tensorflow.compat.v2.data.experimental import (
        choose_from_datasets, AUTOTUNE, sample_from_datasets
)
from tensorflow.keras.applications.resnet50 \
import preprocess_input as preproc_rn
from tensorflow.keras.applications.mobilenet_v2 \
import preprocess_input as preproc_mn
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

tf.compat.v1.enable_eager_execution()

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
        
        resizer = iaa.Sequential([
            iaa.Resize({"height": IMG_SIZE, "width": IMG_SIZE}),
        ])
        augmenter = iaa.Sequential([
            iaa.Resize({"height": IMG_SIZE, "width": IMG_SIZE}),
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.4, 1.6)),
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
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-4, 4)
            )
        ], random_order=True)

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.labels = labels
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.resizer = resizer
        
    def get_pipeline(self, type='train', output='label_bbox', channels=1,
                     apply_aug=True, onehot=True, seed=None, 
                     apply_tl_preprocess=False, model_type="resnet"):
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
            paths_targets = tf.data.Dataset.from_tensor_slices(paths)
            paths_targets = paths_targets.shuffle(BUFFER_SIZE)
            img_targets = paths_targets.map(
                    partial(load_and_resize_image_test, channels=channels),
                    num_parallel_calls=AUTOTUNE)
            img_targets = img_targets.map(standard_scaler_test).repeat()
            ds = img_targets.batch(self.batch_size).prefetch(
                    buffer_size=AUTOTUNE)
            return ds
        else:
            distinct_labels = df['label'].unique()
            #num_labels = len(distinct_labels) # risk of val and train not matching
            num_labels = len(self.labels)
            for car_type in distinct_labels:
                cars = df[df['label']==car_type]
                
                paths = cars['fname']
                labels = cars['label']
                bbox = cars[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]
                
                paths_targets = make_ds(paths.values, 
                                        labels.values, 
                                        bbox.values,
                                        seed=seed)
                
                imgs_targets = paths_targets.map(
                    partial(load_image, channels=channels),
                    num_parallel_calls=AUTOTUNE)
                              
                if apply_aug:
                    imgs_targets = imgs_targets.map(
                            partial(augment_img, 
                                    augmenter=self.augmenter,
                                    output_type=output,
                                    onehot=onehot,
                                    num_labels=num_labels),
                            num_parallel_calls=AUTOTUNE)
                else:
                    imgs_targets = imgs_targets.map(
                            partial(augment_img, 
                                    augmenter=self.resizer,
                                    output_type=output,
                                    onehot=onehot,
                                    num_labels=num_labels),
                            num_parallel_calls=AUTOTUNE)
                            
# =============================================================================
#                 imgs_targets = imgs_targets.map(
#                             partial(log_bbox, 
#                                     output_type=output),
#                             num_parallel_calls=AUTOTUNE)
# =============================================================================

                if not apply_tl_preprocess:
                    imgs_targets = imgs_targets.map(standard_scaler, 
                                                    num_parallel_calls=AUTOTUNE)
                datasets.append(imgs_targets)
            
        num_labels = len(df['label'].unique())
        sampling_weights = np.ones(num_labels)*(1./num_labels)
        
# =============================================================================
#         choice_dataset = tf.data.Dataset.from_tensors([0])
#         choice_dataset = choice_dataset.map(
#                 lambda x: get_random_choice(sampling_weights.tolist()))
#         choice_dataset = choice_dataset.repeat()
# 
#         ds = choose_from_datasets(datasets, choice_dataset)
# =============================================================================
        
        ds = sample_from_datasets(datasets, 
                                  weights=sampling_weights, seed=seed)
        if apply_tl_preprocess:
            ds = ds.map(partial(preprocess, 
                                model_type=model_type),
                        num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        
        return ds    

# =============================================================================
# HELPER PREPROCESS FUNCTIONS
# =============================================================================
     
def preprocess(img, outputs, model_type):
    img = tf.dtypes.cast(img, tf.float32)
    if model_type == "resnet":
        img = tf.numpy_function(preproc_rn, [img], tf.float32)
    elif model_type == "mobilenet":
        img = tf.numpy_function(preproc_mn, [img], tf.float32)
    #img = preprocess_input(img)
    return img, outputs   
    
def make_ds(paths, labels, bbox, seed):
  ds = tf.data.Dataset.from_tensor_slices((paths, 
                                           {"labels": labels, 
                                            "bbox": bbox})).cache()
  ds = ds.shuffle(BUFFER_SIZE, seed=seed).repeat()
  return ds

def get_random_choice(p):
    choice = tf.random.categorical(tf.math.log([p]), 1)
    return tf.cast(tf.squeeze(choice), tf.int64)

def standard_scaler(img, outputs):
    img = tf.cast(img, tf.float16)
    img = img/255.0 - .5
    return img, outputs

def load_image(path, outputs, channels=1):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.cast(img, tf.uint8)
    return img, outputs

def log_bbox(img, outputs, output_type):
    new_outputs = { 'labels': outputs['labels'], 
                    'bbox': tf.math.log(outputs['bbox']) }
    return img, new_outputs

def augment_img(img, outputs, augmenter, output_type, onehot, num_labels):
    
    labels = tf.one_hot(outputs['labels'], num_labels) if onehot \
    else outputs['labels']
    
    def aug_mapper(img, bbox):
        bb_prior = BoundingBoxesOnImage([
            BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        ], shape=img.shape)
        img_aug, bb_after = augmenter(images=[img], bounding_boxes=bb_prior)
        after = bb_after.bounding_boxes[0].clip_out_of_image(img.shape)
        bb_aug = np.asarray([after.x1, after.y1, after.x2, after.y2])
        bb_aug = tf.convert_to_tensor(bb_aug, dtype=tf.float16)
        return img_aug[0], bb_aug
    
    if output_type == 'label':
        img = tf.numpy_function(augmenter.augment_image, [img], tf.uint8)
        new_outputs = { 'labels': labels, 
                       'bbox': outputs['bbox'] }
    else:
        img, bb_aug = tf.numpy_function(aug_mapper, [img, outputs['bbox']], 
                                        (tf.uint8, tf.float16))
        new_outputs = { 'labels': labels, 
                       'bbox': bb_aug }
        
    return img, new_outputs


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



