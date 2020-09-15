from keras.backend import int_shape
from keras.constraints import max_norm
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, add, BatchNormalization, UpSampling2D, LeakyReLU, \
    Reshape, LSTM, Dense
from keras.models import Model

from basemodel.basemodel import BaseModel


class SRDeblur(BaseModel):
    def __init__(self, use_lstm=True):
        super(SRDeblur, self).__init__(f"SRDeblur_cifar_{'lstm' if use_lstm else 'no_lstm'}",
                                       batch_size=150,
                                       epochs=70,
                                       early_stopping_patience=15,
                                       )
        self.use_lstm = use_lstm

    def conv2d(self, before, features, kernel_size=(3, 3), strides=1, activation=None):
        layer = Conv2D(features, kernel_size, strides=strides, padding="same", use_bias=False)(
            before)
        layer = BatchNormalization()(layer)
        if activation is None:
            layer = LeakyReLU()(layer)
        else:
            layer = Activation(activation)(layer)

        return layer

    def deconv2d(self, before, features, kernel_size=(3, 3), strides=1, is_output=False, name_output=None):
        layer = Conv2DTranspose(features, kernel_size, strides=strides, padding="same")(before)
        layer = BatchNormalization()(layer)
        # layer = Activation("relu")(layer) # used in the first experiment
        layer = LeakyReLU()(layer) if is_output else Activation("sigmoid", name=name_output)(layer)
        return layer

    def ResBlock(self, before, features, kernel_size=(3, 3), strides=1):
        # Different from the paper scale recurrent network for deep image deblurring  which use
        # Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper which doesn't use BatchNorm
        conv1 = self.conv2d(before, features, kernel_size, strides)
        conv2 = self.conv2d(conv1, features, kernel_size, strides)
        add1 = add([before, conv2])
        return add1

    def EBlock(self, before, kernel_size=(3, 3), strides=1):
        features = int_shape(before)[-1]
        conv1 = self.conv2d(before, features * 2, kernel_size, strides=2)
        res_block = self.ResBlock(conv1, features * 2, kernel_size, strides)
        res_block = self.ResBlock(res_block, features * 2, kernel_size, strides)
        res_block = self.ResBlock(res_block, features * 2, kernel_size, strides)
        return res_block

    def DBlock(self, before, block_to_connect=None, kernel_size=(3, 3), strides=1):
        features = int_shape(before)[-1]
        if block_to_connect is not None:
            add1 = add([block_to_connect, before])
            res_block = self.ResBlock(add1, features, kernel_size, strides)
        else:
            res_block = self.ResBlock(before, features, kernel_size, strides)
        res_block = self.ResBlock(res_block, features, kernel_size, strides)
        res_block = self.ResBlock(res_block, features, kernel_size, strides)
        # Upsampling reducing the features
        deconv1 = self.deconv2d(res_block, features // 2, kernel_size, 2)
        return deconv1

    def InBlock(self, input, features, kernel_size=(3, 3), strides=1):
        conv1 = self.conv2d(input, features, kernel_size, strides)
        res = self.ResBlock(conv1, features, kernel_size, strides)
        res = self.ResBlock(res, features, kernel_size, strides)
        res = self.ResBlock(res, features, kernel_size, strides)
        return res

    def OutBlock(self, before, output_channel, block_to_connect, features, kernel_size=(3, 3), strides=1,
                 name_output=None):
        add1 = add([block_to_connect, before])
        res = self.ResBlock(add1, features, kernel_size, strides)
        res = self.ResBlock(res, features, kernel_size, strides)
        res = self.ResBlock(res, features, kernel_size, strides)
        # layer for the output image
        shape = int_shape(res)
        w, h = (shape[1], shape[2])
        conv = self.deconv2d(res, output_channel, (1, 1), 1, name_output=name_output)
        return conv

    def ScaleBlock(self, input_layer, output_channel, kernel_size, prev_lstm_state=None, name_output=None):
        strides = 1

        start_features = 32

        in_block = self.InBlock(input_layer, start_features, kernel_size=kernel_size, strides=strides)

        eblock1 = self.EBlock(in_block, kernel_size, strides=strides)

        eblock2 = self.EBlock(eblock1, kernel_size, strides=strides)

        if self.use_lstm:
            prev_shape = list(int_shape(eblock2))
            new_shape = prev_shape[1] * prev_shape[2] * prev_shape[3]
            hidden_features = 32
            lstm = Reshape(target_shape=(1, new_shape))(eblock2)
            # output,
            lstm, h, c = LSTM(hidden_features, return_state=True, kernel_constraint=max_norm(3))(lstm, prev_lstm_state)
            # lstm output = (batch,hidden_features)
            lstm_state = [h, c]
            # Allow to use the same hidden features at different scale
            lstm = Dense(new_shape)(lstm)
            # Restore the correct shape
            lstm = Reshape(target_shape=(prev_shape[1], prev_shape[2], prev_shape[3]))(lstm)

        else:
            lstm_state = None
            lstm = eblock2

        dblock2 = self.DBlock(lstm,
                              kernel_size=kernel_size, strides=strides)

        dblock1 = self.DBlock(dblock2, eblock1, kernel_size, strides)

        out_block = self.OutBlock(dblock1, output_channel, block_to_connect=in_block, features=32,
                                  name_output=f"{name_output}_output")

        return out_block, lstm_state

    def _set_model(self):
        input1 = Input((16, 16, 3))

        kernel_size = (5, 5)

        output1, lstm_state = self.ScaleBlock(input1, 3, kernel_size=kernel_size,
                                              name_output=f"scale{int_shape(input1)[1]}")
        output1_up = UpSampling2D(interpolation="bilinear")(output1)

        input2 = Input((32, 32, 3))

        add1 = add([input2, output1_up])
        output2, lstm_state = self.ScaleBlock(add1, 3, kernel_size=kernel_size, prev_lstm_state=lstm_state,
                                              name_output=f"scale{int_shape(input2)[1]}")

        self._model = Model([input1, input2], [output1, output2])

    def mse_all_axis(self, y_true, y_pred):
        import keras.backend as K
        return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])  # axis=None mean over all axis

    def set_custom_objects(self):
        super(SRDeblur, self).set_custom_objects()
        self.custom_objects.update({"mse_all_axis": self.mse_all_axis})

    def compile(self):
        from keras.optimizers import Adam
        from src.basemodel.metrics import metrics
        # Keras will make a mean of the sum of the two euclidean distance
        if self._model is None: self._set_model()
        compile_args = {"optimizer": Adam(), "loss": [self.mse_all_axis, self.mse_all_axis], "loss_weights": [1, 1],
                        "metrics": metrics}
        self._model.compile(**compile_args)

    def _data(self):
        from src.basemodel.generator.cifar10_generator import cifar10_generator_two_inputs
        from keras.datasets import cifar10
        from sklearn.model_selection import train_test_split

        (X_train, _), (X_test, _) = cifar10.load_data()
        X_train = X_train / 255.
        X_test = X_test / 255.

        validation_split = 0.1
        X_train, X_val = train_test_split(X_train, test_size=validation_split)

        train_gen = cifar10_generator_two_inputs(X_train, self.batch_size, True)
        val_gen = cifar10_generator_two_inputs(X_val, self.batch_size, True)

        test_gen = cifar10_generator_two_inputs(X_test, self.batch_size, False)

        return (train_gen, val_gen, test_gen)

    def test(self):
        import matplotlib.pyplot as plt
        from random import randint
        (_, _, test_gen) = self._data()
        batch_index = randint(0, test_gen.__len__())
        batch_index = 30

        [batch_res_blur, batch_blur], [batch_res, batch] = test_gen.__getitem__(batch_index)
        [y1, y2] = self._model.predict([batch_res_blur, batch_blur])
        n = 3
        fig, axes = plt.subplots(n, 3)

        for i in range(n):
            axes[i][0].imshow(batch_blur[i])
            axes[i][0].set_title("Input")
            axes[i][1].imshow(y2[i])
            axes[i][1].set_title("Predicted image")
            axes[i][2].imshow(batch[i])
            axes[i][2].set_title("True image")

        plt.savefig(self.model_dir / "test.pdf", bbox_inches="tight")
