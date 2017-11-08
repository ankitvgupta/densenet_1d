""" Implementatiions of various DenseNet blocks for 1D sequences"""
from keras.layers import BatchNormalization, Activation, Conv1D, Concatenate, AveragePooling1D

# A single convolutional operation, defined as H_l in the original paper
def H_l(k, bottleneck_size, kernel_width):
    use_bottleneck = bottleneck_size > 0
    num_bottleneck_output_filters = k*bottleneck_size
    def f(x):
        if use_bottleneck:
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(num_bottleneck_output_filters, 1, strides=1, padding="same", dilation_rate=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(k, kernel_width, strides=1, padding="same", dilation_rate=1)(x)
        return x
    return f


def dense_block(k, num_layers, kernel_width, bottleneck_size):

    def f(x):
        layers_to_concat = [x]
        for _ in range(num_layers):
            x = H_l(k, bottleneck_size, kernel_width)(x)
            layers_to_concat.append(x)
            x = Concatenate(axis=-1)(layers_to_concat)
        return x
    return f

def transition_block(pool_size=2, stride=2, theta=0.5):
    assert theta > 0 and theta <= 1
    def f(x):
        num_transition_output_filters = int(int(x.shape[2])*float(theta))
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(num_transition_output_filters, 1, strides=1, padding="same", dilation_rate=1)(x)
        x = AveragePooling1D(pool_size=pool_size, strides=stride, padding="same")(x)
        return x
    return f

