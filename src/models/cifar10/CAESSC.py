from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, add, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.regularizers import l2

from basemodel.basemodel import BaseModel


class CAESSC(BaseModel):
    """
    Convolutional Auto Encoders with Symmetric Skip Connections
    """

    def __init__(self, depth, start_features, downsample):
        """
        :param start_features: amount of filters to use
        :param skip_connections: amount of skip connections
        :param downsample: half the size of the input
        """
        batch_size = 200 if depth == 30 and start_features == 128 else 250
        super(CAESSC, self).__init__(f"CAESSC_d{depth}_f{start_features}{'_half' if downsample else ''}",
                                     batch_size=batch_size, epochs=60,
                                     early_stopping_patience=7)
        self.start_features = start_features
        self.depth = depth
        self.downsample = downsample

    def conv2d(self, before, features, kernel_size=(3, 3), strides=1, name=None):
        out = Conv2D(features, kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name}_conv")(
            before)
        out = BatchNormalization(name=f"{name}_bn")(out)
        out = LeakyReLU(name=f"{name}_lrelu")(out)
        return out

    def conv2d_transpose(self, before, features, kernel_size=(3, 3), strides=1, name=None, use_sigmoid=False):
        out = Conv2DTranspose(features, kernel_size, strides=strides, padding="same", use_bias=False,
                              name=f"{name}_deconv")(before)
        out = BatchNormalization(name=f"{name}_bn")(out)
        out = LeakyReLU(name=f"{name}_lrelu")(out) if not use_sigmoid else Activation("sigmoid",
                                                                                      name=f"{name}_sigmoid")(out)
        return out

    def sum(self, layers):
        out = add(layers)
        out = BatchNormalization()(out)
        out = LeakyReLU()(out)
        return out

    def _set_model(self):
        input_shape = (32, 32, 3)
        features = self.start_features
        depth = self.depth
        kernel_size = (5, 5)
        interval = 2
        n = depth // 2

        # LIFO list
        res_layers = []

        input_layer = Input(shape=input_shape)
        layer = input_layer

        for i in range(1, n + 1):
            if i == 1 and self.downsample:
                layer = self.conv2d(layer, features, kernel_size=kernel_size, strides=2, name="down")
            else:
                layer = self.conv2d(layer, features, kernel_size=kernel_size, name=f"encoder_{i}")
            if i != n and i != 1 and i % interval == 0:
                res_layers.append(layer)

        for i in reversed(range(1, n + 1)):
            # transform i from 0-n-1 to 1-n
            if i != n and i != 1 and i % interval == 0:
                selected_layer = res_layers.pop(-1)
                layer = self.sum([selected_layer, layer])

            if i == 2 and self.downsample:
                # avoid chessboard effect with (4,4)
                layer = self.conv2d_transpose(layer, features, kernel_size=(4, 4), strides=2, name="up")
            elif i == 1:
                # Transform to 3 channels
                layer = self.conv2d_transpose(layer, 3, kernel_size=kernel_size, strides=1, name="output", use_sigmoid=True)
            else:
                layer = self.conv2d_transpose(layer, features, kernel_size=kernel_size, name=f"decoder_{i}")

        output = add([input_layer, layer])

        self._model = Model(input_layer, output)

    def _data(self):
        from src.basemodel.generator.cifar10_generator import cifar10_generators
        (train_gen, val_gen) = cifar10_generators('train', batch_size=self.batch_size)
        test_gen = cifar10_generators('test', batch_size=self.batch_size)

        return (train_gen, val_gen, test_gen)

    def compile(self):
        from keras.optimizers import Adam
        from src.basemodel.metrics import metrics

        if self._model is None: self._set_model()
        compile_args = {"optimizer": Adam(), "loss": "mse", "metrics": metrics}
        self._model.compile(**compile_args)
