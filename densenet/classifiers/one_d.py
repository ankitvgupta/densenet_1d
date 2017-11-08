import keras.models
from keras.layers import Input, Dense
import densenet.models.one_d

class DenseNet121(keras.models.Model):
    """  Create a Keras Model Object that is an implementation of DenseNet121 """

    def __init__(
            self,
            input_shape,
            num_outputs,
            k,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride):
        model_input = Input(shape=input_shape)
        output = densenet.models.one_d.DenseNet121(
            k,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride,
            use_global_pooling=True)(model_input)
        output = Dense(num_outputs, activation="softmax")(output)
        super(DenseNet121, self).__init__(model_input, output)

class DenseNet169(keras.models.Model):
    """ Create a Keras Model Object that is an implementation of DenseNet169 """

    def __init__(
            self,
            input_shape,
            num_outputs,
            k,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride):
        model_input = Input(shape=input_shape)
        output = densenet.models.one_d.DenseNet169(
            k,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride,
            use_global_pooling=True)(model_input)
        output = Dense(num_outputs, activation="softmax")(output)
        super(DenseNet169, self).__init__(model_input, output)


class DenseNet201(keras.models.Model):
    """ Create a Keras Model Object that is an implementation of DenseNet201 """

    def __init__(
            self,
            input_shape,
            num_outputs,
            k,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride):
        model_input = Input(shape=input_shape)
        output = densenet.models.one_d.DenseNet201(
            k,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride,
            use_global_pooling=True)(model_input)
        output = Dense(num_outputs, activation="softmax")(output)
        super(DenseNet201, self).__init__(model_input, output)

class DenseNet264(keras.models.Model):
    """ Create a Keras Model Object that is an implementation of DenseNet264"""

    def __init__(
            self,
            input_shape,
            num_outputs,
            k,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride):
        model_input = Input(shape=input_shape)
        output = densenet.models.one_d.DenseNet264(
            k,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride,
            use_global_pooling=True)(model_input)
        output = Dense(num_outputs, activation="softmax")(output)
        super(DenseNet264, self).__init__(model_input, output)

class DenseNetCustom(keras.models.Model):
    """  Create a Keras Model Object that is an implementation of DenseNet with a custom number of parameters. The number of layers per dense block can be specified by block_sizes."""

    def __init__(
            self,
            input_shape,
            num_outputs,
            k,
            block_sizes,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride):
        model_input = Input(shape=input_shape)
        output = densenet.models.one_d.DenseNet(
            k,
            block_sizes,
            conv_kernel_width,
            bottleneck_size,
            transition_pool_size,
            transition_pool_stride,
            theta,
            initial_conv_width,
            initial_stride,
            initial_filters,
            initial_pool_width,
            initial_pool_stride,
            use_global_pooling=True)(model_input)
        output = Dense(num_outputs, activation="softmax")(output)
        super(DenseNet264, self).__init__(model_input, output)

