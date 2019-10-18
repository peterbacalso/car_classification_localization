# Stanford Cars Dataset Image Classification and Localization

## Methodology

### Model Design

Input Pipeline

The data has a class imbalance so one way to handle this is through oversampling which is creating copies of our minority classes to match the majority ones. Fortunately we do not have to do this explicitly since we can get it for free by modifying the way we output images from the tf.data generator. Oversampling is achieved by splitting the data by their class labels and sampling from them uniformly. This preserves the underlying distribution of the minority classes but evens out the dataset without needing to collect more data!

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/data_pipeline.jpg)

The following is the imbalanced distribution of classes

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/class_imbalance.png)

The following is an example of the distribution after oversampling. The generator was tested for 300 iterations @ 32 batch size each

![](https://github.com/peterbacalso/Cars_Image_Classification_Localization/blob/master/assets/oversampled.png)

## Dataset Citation

Krause, Jonathan, et al. "3d object representations for fine-grained categorization." Proceedings of the IEEE International Conference on Computer Vision Workshops. 2013.
