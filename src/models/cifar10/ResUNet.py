from typing import Any

from keras.backend import int_shape
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import Input
from keras.layers import concatenate, Conv2DTranspose, add
from keras.models import Model

from basemodel.basemodel import BaseModel


class ResUNet(BaseModel):

    def __init__(self, n_resblock=1):
        super(ResUNet, self).__init__(f"ResUNet{n_resblock}", batch_size=250, epochs=50, early_stopping_patience=7)
        self.n_resblock = n_resblock

    def Conv2D(self, before, features,
               kernel_size=(3, 3),
               strides=1,
               activation: Any = "relu",
               ):
        start = Conv2D(features, kernel_size, strides=strides, padding="same", use_bias=False)(before)
        then = BatchNormalization()(start)
        end = Activation(activation)(then) if isinstance(activation, str) else activation(then)
        return end

    def ResBlock(self, before, features, kernel_size=(3, 3)):
        block = self.Conv2D(before, features, kernel_size=kernel_size)
        block = self.Conv2D(block, features, kernel_size=kernel_size)
        add1 = add([before, block])
        return add1

    def UnitBlock(self, n_resblock, before, kernel_size=(3, 3), descending=True, ):
        features = int_shape(before)[-1]
        layer = before
        if descending:
            for _ in range(n_resblock):
                layer = self.ResBlock(layer, features, kernel_size)
            layer = self.Conv2D(layer, features, kernel_size)
        else:
            layer = self.Conv2D(layer, features, kernel_size)
            for _ in range(n_resblock):
                layer = self.ResBlock(layer, features, kernel_size)
        return layer

    def ConcatBlock(self, desc_block_l, asc_block_l_minus_1, kernel_size=(3, 3)):
        features = int_shape(asc_block_l_minus_1)[-1]
        up_block = Conv2DTranspose(features // 2, kernel_size, strides=2, padding="same")(asc_block_l_minus_1)
        conc1 = concatenate([desc_block_l, up_block])
        layer = self.Conv2D(conc1, features // 2, kernel_size)
        return layer

    def _set_model(self):
        input = Input(shape=(32, 32, 3))

        kernel_size = (3, 3)
        features = 32

        desc1 = self.Conv2D(input, features, kernel_size)
        desc1 = self.UnitBlock(self.n_resblock, desc1, kernel_size)

        desc2 = self.Conv2D(desc1, features * 2, kernel_size, strides=2)
        desc2 = self.UnitBlock(self.n_resblock, desc2, kernel_size)

        desc3 = self.Conv2D(desc2, features * 4, kernel_size, strides=2)
        desc3 = self.UnitBlock(self.n_resblock, desc3, kernel_size)

        asc2 = self.ConcatBlock(desc2, desc3, kernel_size)
        asc2 = self.UnitBlock(self.n_resblock, asc2, kernel_size, descending=True)

        asc1 = self.ConcatBlock(desc1, asc2, kernel_size)
        asc1 = self.UnitBlock(self.n_resblock, asc1, kernel_size, descending=True)

        output = Conv2DTranspose(features, kernel_size, activation="relu", padding="same")(asc1)
        output = Conv2DTranspose(3, kernel_size, activation="sigmoid", padding="same")(output)

        model = Model(input, output)

        self._model = model

    def _data(self):
        from basemodel.generator.cifar10_generator import cifar10_generators
        (train_gen, val_gen) = cifar10_generators('train', batch_size=self.batch_size)
        test_gen = cifar10_generators('test')

        return (train_gen, val_gen, test_gen)

    def compile(self):
        from keras.optimizers import Adam
        from basemodel.metrics import metrics

        if self._model is None: self._set_model()
        compile_args = {"optimizer": Adam(), "loss": "mse", "metrics": metrics}
        self._model.compile(**compile_args)
