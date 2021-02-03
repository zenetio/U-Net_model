import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, concatenate)
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, axis=(1, 2), epsilon=0.00001):
    """
    Compute mean dice coefficient over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim)
        y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator of dice coefficient.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant add to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """

    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean((dice_numerator)/(dice_denominator))

    return dice_coefficient

def soft_dice_loss(y_true, y_pred, axis=(1, 2), epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.

    Args:
        y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                    shape: (num_classes, x_dim, y_dim)
        y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                    shape: (num_classes, x_dim, y_dim)
        axis (tuple): spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
                      Hint: pass this as the 'axis' argument to the K.sum
                            and K.mean functions.
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_loss (float): computed value of dice loss.     
    """

    dice_numerator = 2. * K.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true**2, axis=axis) + K.sum(y_pred**2, axis=axis) + epsilon
    dice_loss = 1 - K.mean((dice_numerator)/(dice_denominator))

    return dice_loss

def convolution_block(input_layer, n_filters, batch_normalization=False,
                             kernel=(3, 3), activation=None,
                             #padding='valid', strides=(2, 2),
                             instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    #layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    layer = Conv2D(n_filters, kernel)(input_layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def expanding_block(n_filters, pool_size, kernel_size=(2, 2),
                       strides=(2, 2),
                       deconvolution=False):
    if deconvolution:
        return Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=strides)
    else:
        return UpSampling2D(size=pool_size)

def unet_model_2d(loss_function, input_shape=(572, 572, 3),
                  pool_size=(2, 2), n_labels=3,
                  initial_learning_rate=0.00001,
                  deconvolution=False, depth=5, n_base_filters=64,
                  include_label_wise_dice_coefficients=False, metrics=[],
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 2D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (x_size, y_size, n_chanels). The x, y sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 2D UNet Model
    """
    inputs = Input(shape=input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = convolution_block(input_layer=current_layer,
                                          n_filters=n_base_filters * (2 ** layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = convolution_block(input_layer=layer1,
                                          n_filters=n_base_filters * (2 ** layer_depth),
                                          batch_normalization=batch_normalization)
        if layer_depth < depth -1:
            current_layer = MaxPooling2D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
            
    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        up_convolution = expanding_block(pool_size=pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=current_layer.shape[3] // 2)(current_layer)
        diff_shape = (levels[layer_depth][1].shape[1] - up_convolution.shape[1]) // 2, (levels[layer_depth][1].shape[2] - up_convolution.shape[2]) // 2
        crop1 = Cropping2D(cropping=diff_shape)(levels[layer_depth][1])
        concat = concatenate([up_convolution, crop1], axis=-1)
        current_layer = convolution_block(
            n_filters=levels[layer_depth][1].shape[3],
            input_layer=concat, batch_normalization=batch_normalization)
        current_layer = convolution_block(
            n_filters=levels[layer_depth][1].shape[3],
            input_layer=current_layer,
            batch_normalization=batch_normalization)
    # map the number of outputs channels
    final_convolution = Conv2D(n_labels, (1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=initial_learning_rate), 
                  loss=loss_function, 
                  metrics=metrics)
    return model
