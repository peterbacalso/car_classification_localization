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
from efficientnet.tfkeras import preprocess_input as preproc_efn
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

tf.compat.v1.enable_eager_execution()

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
pd.set_option('display.width', 1000)

IMG_SIZE = 224
BUFFER_SIZE = 100000

class DataLoader():
    
    def __init__(self, train_path='cars_train', 
                 test_path='cars_test', devkit='devkit', 
                 batch_size=32, valid_split=.2):
        devkit_path = Path(devkit)
        
        meta = loadmat(devkit_path/'cars_meta.mat')
        train_annos = loadmat(devkit_path/'cars_train_annos.mat')
        test_annos = loadmat(devkit_path/'cars_test_annos_withlabels.mat')
        
        labels = [c for c in meta['class_names'][0]]
        labels = pd.DataFrame(labels, columns=['labels'])
        
        frame = [[i.flat[0] for i in line] 
                for line in train_annos['annotations'][0]]
        
        columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 
                   'label', 'fname']
        
        df = pd.DataFrame(frame, columns=columns)
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
        
        df_test = pd.DataFrame(test_frame, columns=columns)
        df_test['label'] = df_test['label']-1
        df_test['fname'] = [f'{test_path}/{f}' 
               for f in df_test['fname']] #  Appending Path
        
        df_test = df_test.sort_index()
        
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        
        resizer = iaa.Sequential([
            iaa.Resize({"height": IMG_SIZE, "width": IMG_SIZE}),
        ])
        augmenter = iaa.Sequential([
            iaa.Resize({"height": IMG_SIZE, "width": IMG_SIZE}),
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-4, 4)
            ),
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            iaa.SomeOf((0, 5),
                [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 1.0),
                            n_segments=(20, 200)
                        )
                    ),
    
                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
                    iaa.Alpha(
                        factor=(0.2, 0.8),
                        first=iaa.Sharpen(1.0, lightness=2),
                        second=iaa.CoarseDropout(p=0.1, size_px=8),
                        per_channel=.5
                    ),
                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    
                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    
                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.5)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.5), direction=(0.0, 1.0)
                        ),
                    ])),
    
                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                    ),
    
                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05),
                            per_channel=0.2
                        ),
                    ]),
    
                    # Invert each image's channel with 5% probability.
                    # This sets each pixel value v to 255-v.
                    iaa.Invert(0.05, per_channel=True), # invert color channels
    
                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-10, 10), per_channel=0.5),
    
                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    
                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),
    
                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.LinearContrast((0.4, 1.6), per_channel=True),
                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                # do all of the above augmentations in random order
                random_order=True
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
                     apply_aug=True, onehot=True, seed=None, scale=False,
                     tl_preprocess=False, model_type="resnet"):
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
            scale:          Bool that determines whether to scale img pixel
                            values to be between [-0.5, 0.5]. Only works if
                            tl_preprocess is False
            tl_preprocess:  Bool that determines whether to apply transfer
                            learning preprocessing
            model_type:     Model being used for transfer learning
                            Can be 'resnet', 'mobilenet', or 'efn_b3'
                            
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
            

        distinct_labels = df['label'].value_counts()
        #num_labels = len(distinct_labels) # risk of val and train not matching
        num_labels = len(self.labels)
        for car_type in distinct_labels.index:
            cars = df[df['label']==car_type]
            
            paths = cars['fname']
            labels = cars['label']
            bbox = cars[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]
            
            paths_targets = make_ds(paths.values, 
                                    labels.values, 
                                    bbox.values,
                                    output=output,
                                    seed=seed)
            
            datasets.append(paths_targets)
            
        num_labels = len(distinct_labels)
        sampling_weights = np.ones(num_labels)*(1./num_labels)
        
# =============================================================================
#         total_labels = np.sum(distinct_labels)
#         num_labels = len(distinct_labels)
#         sampling_weights = total_labels/num_labels*1/distinct_labels
# =============================================================================
        
        choice_dataset = tf.data.Dataset.from_tensors([0])
        choice_dataset = choice_dataset.map(
                lambda x: get_random_choice(sampling_weights.tolist()))
        choice_dataset = choice_dataset.repeat()

        paths_targets_allclass = choose_from_datasets(datasets, choice_dataset)
        
        imgs_targets = paths_targets_allclass.map(
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
                    
        if not tl_preprocess and scale:
            imgs_targets = imgs_targets.map(standard_scaler, 
                                            num_parallel_calls=AUTOTUNE)
        
# =============================================================================
#         # tf 2.0 only
#         ds = sample_from_datasets(datasets, 
#                                   weights=sampling_weights, seed=seed)
# =============================================================================
        if tl_preprocess:
            imgs_targets = imgs_targets.map(
                    partial(preprocess, model_type=model_type),
                    num_parallel_calls=AUTOTUNE)
        ds = imgs_targets.batch(self.batch_size).prefetch(buffer_size=AUTOTUNE)
        
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
    elif model_type == "efn_b3":
        img = tf.numpy_function(preproc_efn, [img], tf.float32)
    #img = preprocess_input(img)
    return img, outputs   
    
def make_ds(paths, labels, bbox, output, seed):
    if output == "label_bbox":
        ds = tf.data.Dataset.from_tensor_slices((paths, 
                                                 {"labels": labels, 
                                                  "bbox": bbox})).cache()
    elif output == "label":
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)).cache()
    elif output == "bbox":
        ds = tf.data.Dataset.from_tensor_slices((paths, bbox)).cache()
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
    
    if output_type == 'label_bbox':
        labels = tf.one_hot(outputs['labels'], num_labels) if onehot \
        else outputs['labels']
    elif output_type == 'label':
        labels = tf.one_hot(outputs, num_labels) if onehot else outputs
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
        new_outputs = labels
    elif output_type == "bbox":
        img, bb_aug = tf.numpy_function(aug_mapper, [img, outputs['bbox']], 
                                        (tf.uint8, tf.float16))
        new_outputs = bb_aug
    elif output_type == "label_bbox":
        img, bb_aug = tf.numpy_function(aug_mapper, [img, outputs['bbox']], 
                                        (tf.uint8, tf.float16))
        new_outputs = { 'labels': labels, 
                       'bbox': bb_aug }
        
    return img, new_outputs

