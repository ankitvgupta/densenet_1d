import keras.models
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import densenet.models.one_d


class DenseNet121(keras.models.Model):
    """ Create a Keras Model Object that is a customized Densenet121 implementation"""

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
        #output = Flatten()(output)
        output = Dense(num_outputs, activation="softmax")(output)
        super(DenseNet121, self).__init__(model_input, output)
