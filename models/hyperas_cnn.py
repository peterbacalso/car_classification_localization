import os
import time
import sys 
sys.path.insert(0, os.path.abspath('..'))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, MaxPooling2D, 
    Flatten, Dropout, BatchNormalization,
    Activation, AveragePooling2D, GlobalAvgPool2D
)
from layers.Residual import Residual
from tensorflow.keras.callbacks import (
        TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from functools import partial

from hyperopt import STATUS_OK
from hyperas.distributions import choice, uniform



def model(train_gen, valid_gen):
    
    n_classes = 196
    steps_per_epoch = 204
    validation_steps = 51
    channels=3
    output="label_bbox"
    
    # Tensorboard
    root_logdir = os.path.join(os.curdir, 'logs')
    run_id = time.strftime(f"run_%Y_%m_%d-%H_%M_%S")
    run_logdir = os.path.join(root_logdir, run_id)
    tensorboard = TensorBoard(run_logdir)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
    
     # Model Checkpoints
    checkpoint = ModelCheckpoint(
        filepath=f'checkpoints/' + \
        'epoch.{epoch:02d}_val_loss.{val_loss:.6f}.h5', 
        verbose=1, save_best_only=True)
    
    #Reduce LR on Plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.3,
                                  patience=5, 
                                  min_lr=1e-4, 
                                  verbose=1)
    
    callbacks = [tensorboard, early_stopping, checkpoint, reduce_lr]
    
    DefaultConv2D = partial(Conv2D,
                            kernel_size=3,
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2({{uniform(1e-5, 1)}}),
                            padding="same")
    
    input = Input(shape=(224,224,channels))
    
# =============================================================================
#     conv_1a = DefaultConv2D(filters=64, 
#                             kernel_size=5, 
#                             strides=2)(input)
#     norm_1a = BatchNormalization()(conv_1a)
#     relu_1a = Activation(activation="relu")(norm_1a)
#     
#     max_pool_1 = MaxPooling2D(pool_size=2)(relu_1a)
#     
#     conv_1b = DefaultConv2D(filters=64)(max_pool_1)
#     norm_1b = BatchNormalization()(conv_1b)
#     relu_1b = Activation(activation="relu")(norm_1b)
#     conv_1c = DefaultConv2D(filters=64)(relu_1b)
#     norm_1c = BatchNormalization()(conv_1c)
#     relu_1c = Activation(activation="relu")(norm_1c)
#     
#     max_pool_2 = MaxPooling2D(pool_size=2)(relu_1c)
#     
#     conv_2a = DefaultConv2D(filters=128)(max_pool_2)
#     norm_2a = BatchNormalization()(conv_2a)
#     relu_2a = Activation(activation="relu")(norm_2a)
#     conv_2b = DefaultConv2D(filters=128)(relu_2a)
#     norm_2b = BatchNormalization()(conv_2b)
#     relu_2b = Activation(activation="relu")(norm_2b)
#     
#     max_pool_3 = MaxPooling2D(pool_size=2)(relu_2b)
#     
#     flatten = Flatten()(max_pool_3)
# =============================================================================

    choice_activation = {{choice(['relu', 'leaky_relu'])}}
    
    conv = DefaultConv2D(filters=64, strides=2)(input)
    norm = BatchNormalization()(conv)
    act = Activation(activation=choice_activation)(norm)
    conv = DefaultConv2D(filters=64)(act)
    norm = BatchNormalization()(conv)
    act = Activation(activation=choice_activation)(norm)
    
    choice_pool = {{choice(['max', 'avg'])}}
    if choice_pool == 'max':
        x = MaxPooling2D(pool_size=2)(act)
    else:
        x = AveragePooling2D(pool_size=2)(act)

    for filters in [64] * 3:
        x = Residual(filters, activation=choice_activation)(x)
        
    res_block_depth_1 = {{choice([2, 3])}}
    x = Residual(128, strides=2, activation=choice_activation)(x)
    for filters in [128] * res_block_depth_1:
        x = Residual(filters, activation=choice_activation)(x)
    
    res_block_depth_2 = {{choice([4, 5])}}
    x = Residual(256, strides=2, activation=choice_activation)(x)
    for filters in [256] * res_block_depth_2:
        x = Residual(filters, activation=choice_activation)(x)
        
    x = Residual(512, strides=2, activation=choice_activation)(x)
    for filters in [512] * 2:
        x = Residual(filters, activation=choice_activation)(x)
        
    x = GlobalAvgPool2D()(x)
    
    drop = Dropout({{uniform(0, .5)}})(x)
    dense = Dense(512)(drop)
    norm = BatchNormalization()(dense)
    x = Activation(activation=choice_activation)(norm)
    
    # If we choose 'one', add an additional dense layer
    choice_dense = {{choice(['yes', 'no'])}}
    if choice_dense == 'yes':
        drop = Dropout({{uniform(0, .5)}})(x)
        dense = Dense(256)(drop)
        norm = BatchNormalization()(dense)
        x = Activation(activation=choice_activation)(norm)

    drop = Dropout({{uniform(0, .5)}})(x)
    class_output = Dense(n_classes, 
                         activation="softmax", 
                         name="labels")(drop)
    bbox_output = Dense(units=4, name="bbox")(drop)
    
    # Optimizer
    sgd = SGD(lr={{uniform(.0001, .3)}})
    adam = Adam(lr={{uniform(.0001, .3)}})

    
    choice_opt = {{choice(['adam', 'sgd'])}}
    if choice_opt == 'adam':
        optimizer = adam
    else:
        optimizer = sgd
    

    model=None
    if output=="bbox":
        model = Model(inputs=input, outputs=bbox_output)
        model.compile(loss="msle",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        
    elif output=="label":
        model = Model(inputs=input, outputs=class_output)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
    else:
        model = Model(inputs=input, outputs=[class_output,bbox_output])
        model.compile(loss=["categorical_crossentropy", "msle"],
                      loss_weights=[.8,.2],
                      optimizer=optimizer,
                      metrics=["accuracy"])
    model.fit(
        train_gen,
        epochs=400,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2)
    
# =============================================================================
#     #print(model.metrics_names)
#     # ['loss', 'labels_loss', 'bbox_loss', 'labels_acc', 'bbox_acc']
#     validation_loss = np.amin(result.history['loss']) 
#     print('Best validation loss of epoch:', validation_loss)
# =============================================================================
    
    results = model.evaluate_generator(valid_gen, 
                                       steps=validation_steps,
                                       verbose=0)
    validation_loss = results[0]
    print('val_labels_acc', results[3])
    print('val_bbox_acc', results[4])
    print('Validation Loss:', validation_loss)
    
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}


