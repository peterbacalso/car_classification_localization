from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import efficientnet.tfkeras as efn 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
        Dense, Dropout, GlobalAvgPool2D, BatchNormalization, Activation
        )
import sys, os; 
sys.path.insert(0, os.path.abspath('..'));
from loss_functions.focal_loss import focal_loss

def TL(n_classes, num_frozen_layers, optimizer_type="sgd", 
        lr=.001, reg=1e-6, dropout_chance=0.2, use_focal_loss=False,
        channels=1, output="label_bbox", model_type="resnet"):
    
    if model_type == "resnet":
        base = ResNet50(weights="imagenet", include_top=False,
                        input_shape=(224,224,channels))
    elif model_type == "mobilenet":
        base = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(224,224,channels))
    elif model_type == "efn_b3":
        base = efn.EfficientNetB3(weights="imagenet", include_top=False,
                                  input_shape=(224,224,channels))
        
    x = GlobalAvgPool2D()(base.output)
    
# =============================================================================
#     drop = Dropout(dropout_chance)(x)
#     dense = Dense(units=512, kernel_regularizer=l2(reg))(drop)
#     norm = BatchNormalization()(dense)
#     x = Activation(activation="relu")(norm)
# =============================================================================
    
    drop = Dropout(dropout_chance)(x)
    class_output = Dense(n_classes, 
                         kernel_regularizer=l2(reg),
                         activation="softmax",
                         name="labels")(drop)
    bbox_output = Dense(units=4, name="bbox")(drop)
    
    for i, layer in enumerate(base.layers):
        if i < num_frozen_layers:
            layer.trainable=False
        #print(i, layer)
    
    # Optimizer
    if optimizer_type == "sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01)
    elif optimizer_type == "nesterov_sgd":
        optimizer = SGD(lr=lr, momentum=0.9, decay=0.01, nesterov=True)
    elif optimizer_type == "adam":
        optimizer = Adam(lr=lr)

    model=None
    if output=="bbox":
        model = Model(inputs=base.input, outputs=bbox_output)
        model.compile(loss="msle",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        
    elif output=="label":
        loss = focal_loss(alpha=1) if use_focal_loss \
        else "categorical_crossentropy"
        model = Model(inputs=base.input, outputs=class_output)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=["accuracy"])
    else:
        loss = focal_loss(alpha=1) if use_focal_loss \
        else "categorical_crossentropy"
        model = Model(inputs=base.input, outputs=[class_output,bbox_output])
        model.compile(loss=[loss, "msle"],
                      loss_weights=[.8,.2],
                      optimizer=optimizer,
                      metrics=["accuracy"])
        
    print(model.summary())
    
    return model