from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, add, BatchNormalization
from keras.models import Model

from basemodel.basemodel import BaseModel


class CAESSC(BaseModel):
    """
    Convolutional Auto Encoders with Symmetric Skip Connections
    """

    def __init__(self, depth, start_features, downsample, use_sigmoid=True):
        """
        :param depth: amount of total layers
        :param start_features: amount of filters to use
        :param downsample: half the size of the input
        :param use: if True then Sigmoid is used as last activation function
        """
        batch_size = 300
        if depth >= 30: batch_size = 250
        if depth >= 128: batch_size = 200
        #fixed batch size set for test()
        batch_size = 250
        super(CAESSC, self).__init__(
            f"CAESSC_d{depth}_f{start_features}{'_half' if downsample else ''}{'_no_sigmoid' if not use_sigmoid else ''}",
            batch_size=batch_size, epochs=80, early_stopping_patience=7,last_epoch=0)

        self.start_features = start_features
        self.depth = depth
        self.downsample = downsample
        self.use_sigmoid = use_sigmoid

    def conv2d(self, before, features, kernel_size=(3, 3), strides=1, name=None):
        out = Conv2D(features, kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name}_conv")(
            before)
        out = BatchNormalization(name=f"{name}_bn")(out)
        out = Activation("relu", name=f"{name}_relu")(out)
        return out

    def conv2d_transpose(self, before, features, kernel_size=(3, 3), strides=1, name=None):
        out = Conv2DTranspose(features, kernel_size, strides=strides, padding="same", use_bias=False,
                              name=f"{name}_deconv")(before)
        out = BatchNormalization(name=f"{name}_bn")(out)
        out = Activation("relu", name=f"{name}_relu")(out)
        return out

    def sum(self, layers, use_sigmoid=False):
        out = add(layers)
        out = BatchNormalization()(out)
        out = Activation("relu" if not use_sigmoid else "sigmoid")(out)
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
            if i != n and i != 1 and i % interval == 0:
                layer = self.conv2d_transpose(layer, features, kernel_size=kernel_size, name=f"decoder_{i}")
                selected_layer = res_layers.pop(-1)
                layer = self.sum([selected_layer, layer])
            elif i == 1:
                # Transform to 3 channels
                layer = self.conv2d_transpose(layer, 3, kernel_size=kernel_size, strides=2 if self.downsample else 1,
                                              name="output")
            else:
                layer = self.conv2d_transpose(layer, features, kernel_size=kernel_size, name=f"decoder_{i}")

        output = self.sum([input_layer, layer], use_sigmoid=self.use_sigmoid)

        self._model = Model(input_layer, output)

    def _data(self):
        from basemodel.generator.cifar10_generator import cifar10_generators
        (train_gen, val_gen) = cifar10_generators('train', batch_size=self.batch_size)
        test_gen = cifar10_generators('test', batch_size=self.batch_size)
        return (train_gen, val_gen, test_gen)

    def compile(self):
        from keras.optimizers import Adam
        from basemodel.metrics import metrics

        if self._model is None: self._set_model()
        compile_args = {"optimizer": Adam(learning_rate=1e-4), "loss": "mse", "metrics": metrics}
        self._model.compile(**compile_args)
