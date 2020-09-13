from keras.backend import int_shape
from keras.layers import Conv2D
from keras.layers import Input, Activation, Concatenate, BatchNormalization, Conv2DTranspose
from keras.models import Model

from src.basemodel.basemodel import BaseModel


class EDDenseNet(BaseModel):
    def __init__(self):
        super(EDDenseNet, self).__init__("EDDenseNet", batch_size=65, epochs=50)

    def dense_block(self, x, blocks, features_per_block):
        for i in range(blocks):
            x = self.conv_block(x, features_per_block)
        return x

    def transition_block_down(self, before, reduction):
        layer = BatchNormalization()(before)
        layer = Activation('relu')(layer)
        features = int(int_shape(layer)[-1] * reduction)
        layer = Conv2D(features, (1, 1), use_bias=False)(layer)
        # different from the paper
        # x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        # x = MaxPooling2D(2, strides=2, name=name + '_pool')(x)
        layer = Conv2D(features, (2, 2), strides=2)(layer)
        return layer

    def trasition_block_up(self, before):
        features = int_shape(before)[-1]
        layer = Conv2DTranspose(features, kernel_size=(2, 2), strides=2)(before)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        return layer

    def conv_block(self, before, growth_rate):
        before = BatchNormalization()(before)
        before = Activation('relu')(before)

        # bottlneck
        layer = Conv2D(4 * growth_rate, 1,
                       use_bias=False)(before)

        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False)(layer)
        x = Concatenate(axis=-1)([before, layer])
        return x

    def _set_model(self):
        input_layer = Input((32, 32, 3))
        output_layer = Conv2DTranspose(3, (1, 1), strides=1, padding="same", activation="sigmoid")

        blocks = [6, 6, 3]

        features = 16

        x = Conv2D(features, (3,3), strides=1, padding="same", activation="relu")(input_layer)

        x = self.dense_block(x, blocks[0], features)
        x = self.transition_block_down(x, 0.5)
        x = self.dense_block(x, blocks[1], features)
        x = self.transition_block_down(x, 0.5)
        x = self.dense_block(x, blocks[2], features)

        # bottleneck for reducing features
        x = Conv2D(features, (1, 1), strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        blocks = [3, 6]

        x = self.trasition_block_up(x)
        x = self.dense_block(x, blocks[0], features)
        x = self.trasition_block_up(x)
        x = self.dense_block(x, blocks[1], features)

        self._model = Model(input_layer, output_layer(x))

    def _data(self):
        from src.basemodel.generator.cifar10_generator import cifar10_generators
        (train_gen, val_gen) = cifar10_generators('train', batch_size=self.batch_size)
        test_gen = cifar10_generators('test', batch_size=self.batch_size)

        return (train_gen, val_gen, test_gen)

    def compile(self):
        from keras.optimizers import Adam
        from src.basemodel.metrics import metrics

        if self._model is None:
            self._set_model()
        compile_args = {"optimizer": Adam(), "loss": "mse", "metrics": metrics}
        self._model.compile(**compile_args)
