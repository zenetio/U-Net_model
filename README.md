# The U-Net model

The U-Net model has two main parts: the `contracting path` which is the left side of the model and the `expanding path` which is the right side of the model. The image below shows the U-Net architecture and how it contracts and then expands.
![image](https://drive.google.com/uc?export=view&id=1XgJRexE2CmsetRYyTLA7L8dsEwx7aQZY)

We can see that images are first fed through many convolutional layers which reduce height and width while increasing the channels, which the authors refer to as the contracting path. For example, a set of two 2 x 2 convolutions with a stride of 2, will take a 1 x 28 x 28 (channels, height, width) grayscale image and result in a 2 x 14 x 14 representation. The expanding path does the opposite, gradually growing the image with fewer and fewer channels.

## The Contracting Path

This path is the encoder section of the U-Net, which has several downsampling steps as part of it. The authors give more detail of the remaining parts in the following paragraph from the paper (Renneberger, 2015):

The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3 x 3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2 x 2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels.

The different arrows represent different layers and operations

![contract](images/contract.jpg)

<p>
    <bold><center>Contracting Path</center></bold>
</p>

<p>
    <img src="images/arrows.png" alt>
</p>
<p>
    <bold><center>Arrows label</center></bold>
</p>

The images are first fed through many convolutional layers which reduce height and width while increasing the channels, which the authors refer to as the `contracting path`. For example, a set of two 2 x 2 convolutions with a stride of 2, will take a 1 x 28 x 28 (channels, height, width) grayscale image and result in a 2 x 14 x 14 representation. The `expanding path` does the opposite, gradually growing the image with fewer and fewer channels.

So, consider in the Part-1 of the U-Net architecture, where 64 represents the filter size to be used in the convolutional layer and (572x572) (570x570)(568,568) represents the image size. Here the paper explains the output image size after each layer.
<p>
    <img src="images/part1.png" alt>
</p>
<p>
    <bold><center>Part - 1</center></bold>
</p>
Note that for each conv layer, we have a decrease of 2 in image size. Remember the deep learning class where when we have no padding, the image size is decreased by 2, or 1 pixel x 2 sides.

Then looking to the `contract path` architecture, we have the following sequence: C64-C128-C256-C512-C512-C512-C512-C512, where C# denotes a convolution layer followed by the number of filters.

We can create this sequence using a loop-for over a **convolution block** that we can implement.

## Expanding Path

This is the decoding section of U-Net which has several upsampling steps as part of it.  In order to do this, we will need to use a crop function. So, we can crop the image from the `contracting path` and concatenate it to the current image in the `expanding path`- this is to form a skip connection. Again, the details are from the paper (Renneberger, 2015):

>Every step in the expanding path consists of an upsampling of the feature map followed by a 2 x 2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3 x 3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution.

<p>
    <img src="images/crop.png" alt>
</p>
<p>
    <bold><center>Cropping image</center></bold>
</p>

The `expanding path`architecture is: CD512-CD512-CD512-C512-C256-C128-C64, where CD# denotes a deconvolution layer followed by the number of filters.

Note that we will have to get the output of C512 layer and crop the image from 64x64 to 56x56 so that it can be concatenated with `expanding path` in CD512. For `expanding block` we use the Conv2DTranspose layer that performs an inverse convolution operation.

Now we can implement the expanding block.

## Putting all together

Now that we have both sides of the U architecture, we can put all together and create the U-Net model calling `unet_model_2d()` function.

```{python}
model = unet_model_2d(loss_function=soft_dice_loss, input_shape=(572,572,3), metrics=[dice_coefficient], depth=5, n_labels=4, deconvolution=True)
```

If you want to work with 3D, you can just type

```{python}
model = unet_model_3d(loss_function=soft_dice_loss, metrics=[dice_coefficient])
```

You can see the application of an example in the `Train U-Net.ipynb` notebook. 