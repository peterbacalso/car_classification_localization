import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import time

from IPython.display import clear_output
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from loss_functions.focal_loss import focal_loss
import efficientnet.tfkeras as efn
from efficientnet.tfkeras import preprocess_input as preproc_efn

#tf.compat.v1.enable_eager_execution()

# Download an image and read it into a NumPy array.
def download(path, target_size=None):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    img = np.array(img)
    return img


# Normalize an image
def deprocess(img):
    mean = [0.485, 0.456, 0.406] # efn
    std = [0.229, 0.224, 0.225]
    img = (img*std+mean)*255
    return tf.cast(img, tf.uint8)

# Display an image
def show(img):
    plt.figure(figsize=(12,12))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

@tf.function
def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.

    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    
    return  tf.reduce_sum(losses)

@tf.function
def deepdream(model, img, step_size):
    with tf.GradientTape() as tape:
      # This needs gradients relative to `img`
      # `GradientTape` only watches `tf.Variable`s by default
      tape.watch(img)
      loss = calc_loss(img, model)

    # Calculate the gradient of the loss with respect to the pixels of the input image.
    gradients = tape.gradient(loss, img)

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8 
    
    # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
    # You can update the image by directly adding the gradients (because they're the same shape!)
    img = img + gradients*step_size
    img = tf.clip_by_value(img, -1, 1)

    return loss, img
  
@tf.function
def run_deep_dream_simple(model, img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = preproc_efn(img)
    
    for step in range(steps):
        loss, img = deepdream(model, img, step_size)
    
        if step % 500 == 0:
            clear_output(wait=True)
            show(deprocess(img))
            print ("Step {}, loss {}".format(step, loss))
    
    
    result = deprocess(img)
    clear_output(wait=True)
    show(result)
    
    return result

@tf.function
def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tf.random.uniform(shape=[2], minval=-maxroll, 
                              maxval=maxroll, dtype=tf.int32)
    shift_down, shift_right = shift[0],shift[1] 
    img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
    return shift_down, shift_right, img_rolled

@tf.function
def get_tiled_gradients(model, img, tile_size=512):
    shift_down, shift_right, img_rolled = random_roll(img, tile_size)
    # Initialize the image gradients to zero.
    gradients = tf.zeros_like(img_rolled)
    
    for x in range(0, img_rolled.shape[0], tile_size):
        for y in range(0, img_rolled.shape[1], tile_size):
            # Calculate the gradients for this tile.
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img_rolled`.
                # `GradientTape` only watches `tf.Variable`s by default.
                tape.watch(img_rolled)
                # Extract a tile out of the image.
                img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
                loss = calc_loss(img_tile, model)
            # Update the image gradients for this tile.
            gradients = gradients + tape.gradient(loss, img_rolled)
    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), 
                        -shift_down, axis=0)

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8 

    return gradients

def run_deep_dream_with_octaves(model, img, 
                                steps_per_octave=1000, step_size=0.01, 
                                num_octaves=3, octave_scale=1.3):
    img = preproc_efn(img)
    
    for octave in range(num_octaves):
        # Scale the image based on the octave
        if octave>0:
            new_size = tf.cast(tf.convert_to_tensor(img.shape[:2]), 
                               tf.float32)*octave_scale
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

    for step in range(steps_per_octave):
        gradients = get_tiled_gradients(model, img)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)
        
        if step % 500 == 0:
            clear_output(wait=True)
            show(deprocess(img))
            print ("Octave {}, Step {}".format(octave, step))
    
    clear_output(wait=True)
    result = deprocess(img)
    show(result)

    return result

def load_cnn():
    model = load_model('./old_checkpoints/efn50_62_frozen/epoch.130_val_loss.0.337101.h5', 
                       custom_objects={'focal_loss_fixed': focal_loss(alpha=1)})
    return model

if __name__=="__main__":
    
    df_test = pd.read_csv('./data_tables/test.csv')
    filepath = df_test[df_test['fname'].str.contains("00666")]['fname'].values[0]
    original_img = download(f'./data/{filepath}', target_size=(224, 224))

    # Maximize the activations of these layers
    model = load_cnn()
    
    names = ['block7b_add', 'block6f_add']
    layers = [model.get_layer(name).output for name in names]
    
    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=model.input, outputs=layers)
    
# =============================================================================
#     dream_img = run_deep_dream_simple(model=dream_model, img=original_img, 
#                                       steps=10000, step_size=0.001)
# =============================================================================
    
    dream_img = run_deep_dream_with_octaves(model=dream_model, 
                                            img=original_img, 
                                            step_size=0.01)

    clear_output()
    show(original_img)
    show(dream_img)
        