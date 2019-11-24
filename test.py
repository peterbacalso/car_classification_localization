import os
import random
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Model
        
from data.data_loader import DataLoader

from models.transfer_learning import TL
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from vis.utils import utils

import efficientnet.tfkeras as efn
from efficientnet.tfkeras import preprocess_input as preproc_efn
from loss_functions.focal_loss import focal_loss

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

clear_session() # Clear models from previous sessions


def load_data(output="label_bbox", batch_size=32, channels=1, 
              tl_preprocess=False, model_type="resnet", seed=23):
    
    data = DataLoader('./data/cars_train', 
                  './data/cars_test', 
                  './data/devkit', 
                  batch_size=batch_size)
    n_classes = len(data.df_test['label'].unique())
    labels = data.labels
    print(f'{n_classes} CLASSES, Random Chance: {1/n_classes}')
    
    gen = data.get_pipeline(type='test',
                            output=output,
                            channels=channels,
                            apply_aug=False,
                            tl_preprocess=tl_preprocess,
                            model_type=model_type,
                            seed=seed)
    steps = np.ceil(len(data.df_test)/data.batch_size)

    return labels, n_classes, gen, steps
        
        
def plot_activation(model, img, labels):
    plt.rcParams['figure.figsize'] = (18, 6)
    
    pred = model.predict(img[np.newaxis,:,:,:])[0]
    pred_class = np.argmax(pred)

    weights = model.layers[-2].get_weights()[0]
    class_weights = weights[:, pred_class]

    intermediate = Model(model.input,
                         model.get_layer("top_conv").output)
    conv_output = intermediate.predict(img[np.newaxis,:,:,:])
    conv_output = np.squeeze(conv_output)

    h = int(img.shape[0]/conv_output.shape[0])
    w = int(img.shape[1]/conv_output.shape[1])

    act_maps = sp.ndimage.zoom(conv_output, (h, w, 1), order=1)

    out = np.dot(act_maps.reshape((img.shape[0]*img.shape[1],-1)), 
                 class_weights).reshape(img.shape[0],img.shape[1])

    plt.imshow(img.astype('float32').reshape(img.shape[0],
               img.shape[1],3))
    plt.imshow(out, cmap='jet', alpha=0.35)
    plt.title(labels[pred_class])
    plt.show()
    

def create_confusion_matrix(gen, model, classes, batch_size, steps, n_classes):
    y_preds = np.zeros(shape=(steps*batch_size,n_classes))
    y_true = np.zeros(shape=(steps*batch_size,n_classes))
    i = 0
    for images, outputs in gen.take(steps):
        test_batch = images.numpy()
        y_probs = model.predict(test_batch)
        start = i*batch_size
        end = start + batch_size
        y_preds[start:end] = y_probs[0]
        y_true[start:end] = outputs['labels']
        i += 1
    
    y_preds_top1 = np.argmax(y_preds, axis=1)
    y_true_top1 = np.argmax(y_true, axis=1)
    
    
# =============================================================================
#     y_true = pd.Series(true_data, name="Actual")
#     y_pred = pd.Series(pred_data, name="Predicted")
#     df_confusion = pd.crosstab(y_true, y_pred)
# =============================================================================
    
    cm = plot_confusion_matrix(y_true_top1, y_preds_top1, 
                          classes, title="Cars Classification")
    
    return cm
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    plt.figure(figsize=(100,100))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                     xticklabels=classes, yticklabels=classes)
    ax.set_xticklabels(ax.get_xticklabels(), 
                       rotation=70, 
                       horizontalalignment='right',
                       fontweight='light')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    return cm

def output_test(res, inputs, outputs):
    #mean = [103.939, 116.779, 123.68] # resnet
    mean = [0.485, 0.456, 0.406] # efn
    std = [0.229, 0.224, 0.225]
    plt.figure(figsize=(20, 20))
    for i in range(res[0].shape[0]):
        
        #index = np.argmax(outputs['labels'][i].numpy())
        pred_index = np.argmax(res[0][i])
        confidence = np.max(res[0][i])
        true_index = np.argmax(outputs['labels'][i].numpy())
        pred_title = labels.iloc[pred_index]['labels']
        true_title = labels.iloc[true_index]['labels']
        title = f'T:{true_title}\nP:{pred_title}\nConfidence: {int(confidence*100)}%'
        
        #img = (inputs.numpy()[i] + mean)[..., ::-1] #resnet
        img = (inputs.numpy()[i]*std+mean)*255

        
        bbox_x1 = res[1][i][0]
        bbox_y1 = res[1][i][1]
        bbox_x2 = res[1][i][2]
        bbox_y2 = res[1][i][3]
        
        xy = bbox_x1, bbox_y1
        width = bbox_x2 - bbox_x1
        height = bbox_y2 - bbox_y1
        rect = Rectangle(xy, width, height, 
                         fill=False, color='r', linewidth=2)
        
        t_bbox_x1 = outputs['bbox'][i][0].numpy()
        t_bbox_y1 = outputs['bbox'][i][1].numpy()
        t_bbox_x2 = outputs['bbox'][i][2].numpy()
        t_bbox_y2 = outputs['bbox'][i][3].numpy()
        
        t_xy = t_bbox_x1, t_bbox_y1
        t_width = t_bbox_x2 - t_bbox_x1
        t_height = t_bbox_y2 - t_bbox_y1
        t_rect = Rectangle(t_xy, t_width, t_height, 
                           fill=False, color='b', linewidth=2)
        
        plt.subplot(4,4,i+1)
        plt.imshow(img.astype(np.uint8), aspect='auto')
        plt.axis('off')
        plt.title(title, fontsize=11)
        plt.gca().add_patch(rect)
        plt.gca().add_patch(t_rect)
        
    plt.show()


if __name__=="__main__":
    labels, n_classes, gen, steps = \
    load_data(batch_size=16, 
              channels=3,
              tl_preprocess=True,
              model_type="efn_b3")
    
    #tf.compat.v1.disable_eager_execution()
    
    model = load_model('./old_checkpoints/efn50_62_frozen/epoch.130_val_loss.0.337101.h5', 
                       custom_objects={'focal_loss_fixed': focal_loss(alpha=1)})
    
    # test accuracy
    results = model.evaluate_generator(gen, 
                                       steps=steps,
                                       verbose=1)
    test_loss = results[0]
    print('test_labels_acc', results[3])
    print('test_bbox_acc', results[4])
    print('Test Loss:', test_loss)
    
# =============================================================================
#     # confusion matrix
#     cm = create_confusion_matrix(gen, model, labels['labels'], 
#                                  batch_size=32, 
#                                  steps=int(steps), 
#                                  n_classes=n_classes)
#     
#     recall = np.diag(cm) / np.sum(cm, axis = 1)
#     precision = np.diag(cm) / np.sum(cm, axis = 0)
#     data = {'Recall':recall.round(4), 'Precision':precision.round(4)} 
#     prec_rec = pd.DataFrame(data, index = labels['labels'])
#     
#     # heatmap
#     img1 = utils.load_img('./data/cars_test/01414.jpg', 
#                           target_size=(224, 224)) # 159
#     img2 = utils.load_img('./data/cars_test/00200.jpg', 
#                           target_size=(224, 224)) # 159    
#     for img in [img1, img2]:
#         plot_activation(model, preproc_efn(img), labels['labels'])
#     
#     # visualize 16 predictions
#     for inputs, outputs in gen.take(1):
#         res = model.predict(inputs)
#         output_test(res, inputs, outputs)
# =============================================================================



