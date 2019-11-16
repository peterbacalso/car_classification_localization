import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model

from data.data_loader import DataLoader

import random
from PIL import Image
from matplotlib.patches import Rectangle

clear_session() # Clear models from previous sessions


def load_data(output="label_bbox", batch_size=32, channels=1, 
              apply_tl_preprocess=False, model_type="resnet", seed=23):
    
    data = DataLoader('./data/cars_train', 
                  './data/cars_test', 
                  './data/devkit', 
                  batch_size=batch_size)
    n_classes = len(data.df_train['label'].unique())
    labels = data.labels
    print(f'{n_classes} CLASSES, Random Chance: {1/n_classes}')
    
    
    gen = data.get_pipeline(type='validation',
                                  output=output,
                                  channels=channels,
                                  apply_aug=False,
                                  apply_tl_preprocess=apply_tl_preprocess,
                                  model_type=model_type,
                                  seed=seed)
    steps = np.ceil(len(data.df_valid)/data.batch_size)

    return labels, n_classes, gen, steps
        
        
    


if __name__=="__main__":
    
    labels, n_classes, gen, steps = \
    load_data(batch_size=32, 
              channels=3,
              apply_tl_preprocess=True,
              model_type="monbilenet")
    
    #model = load_model('./old_checkpoints/mn_37_frozen/epoch.71_val_loss.0.849966.h5')
    model = load_model('./old_checkpoints/rn50_123_frozen/epoch.49_val_loss.0.846291.h5')
    
    #print(model.summary())
    predictions = None
    valid_inputs = None
    valid_outputs = None
    for inputs, outputs in gen.take(1):
        valid_inputs = inputs
        valid_outputs = outputs
        predictions = model.predict(inputs)
        
    def output_test(res, outputs):
        for i in range(res[0].shape[0]):
            
            #index = np.argmax(outputs['labels'][i].numpy())
            pred_index = np.argmax(res[0][i])
            confidence = np.max(res[0][i])
            true_index = np.argmax(outputs['labels'][i].numpy())
            pred_title = labels.iloc[pred_index]['labels']
            true_title = labels.iloc[true_index]['labels']
            title = f'T/P:{true_title}/{pred_title}, {confidence}'
            
            img = inputs.numpy()[i]
            
            #image = Image.fromarray(img)
            plt.imshow(img.astype(np.uint8))
            
            bbox_x1 = res[1][i][0]
            bbox_y1 = res[1][i][1]
            bbox_x2 = res[1][i][2]
            bbox_y2 = res[1][i][3]
            
            xy = bbox_x1, bbox_y1
            width = bbox_x2 - bbox_x1
            height = bbox_y2 - bbox_y1
            rect = Rectangle(xy, width, height, 
                             fill=False, color='r', linewidth=2)
            
            plt.axis('off')
            plt.title(title)
            plt.gca().add_patch(rect)
            
            plt.show()
       











# HYPERPARAMS

# =============================================================================
# params = {
#         "lr":  [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
#         #"reg": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
#         "reg": [1e-3, 1e-2, 1e-1]
# }
# =============================================================================
     

# TESTING

# =============================================================================
# test_gen_clf = data.get_pipeline(type='test', output='label', apply_aug=False)
# # test_gen_clf_localize = data.get_pipeline(type='test', apply_aug=False)
# steps=tf.math.ceil(len(data.df_test)/data.batch_size)
# tf.cast(steps, tf.int16).numpy()
# 
# index_to_label = data.labels.to_dict()['labels']
# 
# # Display Test Predictions
# def index_to_class(index, classes):
#     return classes[index]
# 
# def decode_predictions(y_probs, classes, top=3):
#     top_probs = np.sort(y_probs, axis=1)[:, ::-1][:, :top] # desc
#     top_indeces = np.argsort(y_probs, axis=1)[:, ::-1][:, :top] # desc
#     vfunc = np.vectorize(index_to_class)
#     top_cat = vfunc(top_indeces, classes)
#     return [top_probs, top_cat]
# 
# for images in test_gen_clf.take(1):
#     test_batch = images.numpy()
#     results = clf_model.predict(test_batch)
#     top_k = 3
#     top_k_probs, top_k_labels = decode_predictions(results, index_to_label, top=top_k)
#     plt.figure(figsize=(3,3))
#     for i in range(BATCH_SIZE):
#         print(f"Image #{i+1}")
#         img = test_batch[i]
#         img.resize(224,224)
#         plt.imshow(img)
#         plt.show()
#         for j in range(top_k):
#             print("    {:12s} {:.2f}%".format(top_k_labels[i][j], top_k_probs[i][j] * 100))
#         print()
# =============================================================================

# =============================================================================
# for images in test_gen_clf_localize.take(1):
#     test_batch = images.numpy()
#     results = clf_localize_model.predict(test_batch)
#     top_k = 3
#     top_k_probs, top_k_labels = decode_predictions(results[0], index_to_label, top=top_k)
#     plt.figure(figsize=(3,3))
#     for i in range(BATCH_SIZE):
#         print(f"Image #{i+1}")
#         img = test_batch[i]
#         img.resize(224,224)
#         plt.imshow(img)
#         plt.show()
#         for j in range(top_k):
#             print("    {:12s} {:.2f}%".format(top_k_labels[i][j], top_k_probs[i][j] * 100))
#         print()
# =============================================================================
# =============================================================================
# model.evaluate(test_generator, steps=steps)
# =============================================================================