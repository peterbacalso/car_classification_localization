# Stanford Cars Dataset Image Classification and Localization

TODO: Add image examples

# Input Pipeline

For preprocessing, making the data zero-centered would be computationally expensive to load all images and calculate the mean. Therefore we will just scale the pixels to be in range [-0.5,0.5] and use batch normalization between layers.
When a transfer learning model is used for training, the corresponding `prerpocess_input` function is applied instead.

There are only a few (~30-50) images per class so to combat overfitting, heavy augmentation was applied through the library [imgaug](https://imgaug.readthedocs.io/en/latest/). See [data_loader.py](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/data/data_loader.py) for implementation details.

TODO: Add augmented image examples

The data has a class imbalance so one way to handle this is through oversampling which is creating copies of our minority classes to match the majority ones. Fortunately we do not have to do this explicitly since we can get it for free by modifying the way we output images from the tf.data generator. Oversampling is achieved by splitting the data by their class labels and sampling from them uniformly. This preserves the underlying distribution of the minority classes but evens out the dataset without needing to collect more data!

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/data_pipeline.jpg)

The following is the imbalanced distribution of classes

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/class_imbalance.png)

The following is an example of the distribution after oversampling. The generator was tested for 10000 iterations @ 32 batch size each

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/oversampled_320000.png)

## Sanity Checks

This step was done to help monitor training and adjust hyperparameters to get good learning results.

1. When using softmax, the value of the loss when the weights are small and no regularization is used can be approximated by -ln(1/C) = ln(C) where C is the number of classes.

The entire dataset has 196 classes which means the softmax loss should be approximately ln(196)=5.278. After running one epoch on a neural net with 1 hidden layer, the loss did in fact match.
```
217/217 [==============================] - 17s 80ms/step - loss: 5.2780 - accuracy: 0.0049 - val_loss: 5.2947 - val_accuracy: 0.0032
```
The same process was repeated for a subset of the dataset using 2 labels. The loss should be ln(2)=0.693.
```
3/3 [==============================] - 1s 233ms/step - loss: 0.6933 - accuracy: 0.5625 - val_loss: 0.5985 - val_accuracy: 0.6875
```
2. Adding regularization should make the loss go up. The following test adds l2 regularization of magnitude 1e2 which made the loss jump from 0.693 to 2.9.
```
3/3
 [==============================] - 1s 322ms/step - loss: 2.9040 - accuracy: 0.4375 - val_loss: 2.9195 - val_accuracy: 0.6875
```
## Model Architectures

### Custom Model

Traditional:

1. Conv(64 filters, 5x5 kernel, 2 strides)|BatchNorm|Relu|MaxPool(2 pool size)
2. [Conv(128 filters, 3x3 kernel, 1 strides)|BatchNorm|Relu]*2|MaxPool(2 pool size)
3. [Conv(256 filters, 3x3 kernel, 1 strides)|BatchNorm|Relu]*2|MaxPool(2 pool size)
4. [Drop|Dense(512 units)|BatchNorm|Relu]*2
5. a.) Drop|Dense (196 units)
5. b.) Drop|Dense (4 units)

Residual:

1. Conv(64 filters, 3x3 kernel, 2 strides)|BatchNorm|Relu
2. [Conv(64 filters, 3x3 kernel, 1 strides)|BatchNorm|Relu]*2|MaxPool(2 pool size)
3. Residual(64 filters)*3|Residual(128 filters)*4|Residual(256 filters)*4|Residual(512 filters)*3
4. GlobalAvgPool2D|[Drop|Dense(512 units)|BatchNorm|Relu]*2
5. a.) Drop|Dense (196 units)
5. b.) Drop|Dense (4 units)

### Transfer Learning

- ResNet50
- MobileNetV2
- EfficientNet-B3

## Training Process

1. Train on a small subset of data (eg. 20 samples) which should be easy to overfit and get a high training accuracy. The subset size used for this step was 73 images over 2 classes and ran for 200 epochs that resulted in 100% classifier accuracy.
```
Epoch 197/200
3/3 [==============================] - 3s 1s/step - loss: 0.0555 - classifier_loss: 0.0281 - localizer_loss: 0.1651 - classifier_accuracy: 1.0000 - localizer_accuracy: 0.4271 - val_loss: 15.2211 - val_classifier_loss: 11.0812 - val_localizer_loss: 31.7805 - val_classifier_accuracy: 0.3125 - val_localizer_accuracy: 0.8125
Epoch 198/200
3/3 [==============================] - 3s 1s/step - loss: 0.0425 - classifier_loss: 0.0163 - localizer_loss: 0.1473 - classifier_accuracy: 1.0000 - localizer_accuracy: 0.4479 - val_loss: 15.2499 - val_classifier_loss: 11.0812 - val_localizer_loss: 31.9246 - val_classifier_accuracy: 0.3125 - val_localizer_accuracy: 0.9062
Epoch 199/200
3/3 [==============================] - 3s 1s/step - loss: 0.0487 - classifier_loss: 0.0264 - localizer_loss: 0.1382 - classifier_accuracy: 1.0000 - localizer_accuracy: 0.3542 - val_loss: 15.2735 - val_classifier_loss: 11.0812 - val_localizer_loss: 32.0426 - val_classifier_accuracy: 0.3125 - val_localizer_accuracy: 0.8125
Epoch 200/200
3/3 [==============================] - 3s 1s/step - loss: 0.0572 - classifier_loss: 0.0329 - localizer_loss: 0.1546 - classifier_accuracy: 1.0000 - localizer_accuracy: 0.3958 - val_loss: 15.3056 - val_classifier_loss: 11.0812 - val_localizer_loss: 32.2035 - val_classifier_accuracy: 0.3125 - val_localizer_accuracy: 0.8125
```
2. Train using full dataset, start with small regularization and find the learning rate that makes the loss go down. The model is able to overfit with train accuracy of 1 implying that it has enough capacity to learn the image features.

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/all_class_adam_2.png)

3. Now that we know the model can overfit, we can increase regularization and tune hyperparameters.

Wandb was used for logging all experiments on the full dataset:

- [Training Experiments](https://app.wandb.ai/peterbacalso/car_classification?workspace=user-peterbacalso)

Initially custom models were used for training but these proved to be difficult for finding a good solution. Each experiment was time consuming since the validation loss would converge slowly and the best validation accuracy it achieved was only at 50%.
Swapping the model to a pretrained one based on imagenet dramatically improved both the results and the time it took to reach a decent accuracy.

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/finally_converging.png)

As shown on the image above, the blue line represents the transfer learning model and at epoch 5 it has already reached 65% label accuracy.

TODO: try siames networks

## Results

The best results so far was achieved by EfficientNet-B3 with the following [hyperparameters](https://app.wandb.ai/peterbacalso/car_classification/runs/0p6yeqbq/overview) and using Focal Loss.

TODO: short Focal Loss explanation

Training vs validation metrics were close during model training so underfitting or overfitting did not occur
Validation Loss: 

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/loss.png)

Validation Labels Accuracy:

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/labels_acc.png)

Validation Bounding Box Accuracy:

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/bbox_acc.png)


Activation Heatmap

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/mclaren_heatmap.png)

TODO: 
- Hyperas Training
- Add Confusion matrix (its big how to insert here)
- Try Gradient Ascent (deep dream?)
- Calculate IOU for bbox
- Streamlit link??

## Challenges

## Dataset Citation

Krause, Jonathan, et al. "3d object representations for fine-grained categorization." Proceedings of the IEEE International Conference on Computer Vision Workshops. 2013.
