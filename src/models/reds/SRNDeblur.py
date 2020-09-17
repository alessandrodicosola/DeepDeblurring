import tensorflow as tf
from keras.backend import int_shape
from keras.constraints import max_norm
from keras.layers import Input, Conv2D, Conv2DTranspose, Activation, add, UpSampling2D, ConvLSTM2D, Reshape, concatenate
from keras.models import Model

from basemodel.basemodel import BaseModel


class SRNDeblur(BaseModel):
    def __init__(self, low_res_test):
        super(SRNDeblur, self).__init__("SRDeblur_reds_hr",
                                        batch_size=5,
                                        epochs=150, early_stopping_patience=10,
                                        last_epoch=0,
                                        save_on_interval=2
                                        )
        self.low_res = low_res_test
        self.num_patches = 2
        self.patch_size = (256, 256)
        self.patch_size_reduced = tuple(map(lambda x: x // 2, self.patch_size))

        self.input_size = self.patch_size

        self.original_size = (320, 180, 3) if self.low_res else (1280, 720, 3)

    def conv2d(self, before, features, kernel_size=(3, 3), strides=1, use_activation=True):
        layer = Conv2D(features, kernel_size, strides=strides, padding="same", use_bias=False)(
            before)
        if use_activation: layer = Activation("relu")(layer)
        return layer

    def deconv2d(self, before, features, kernel_size=(3, 3), strides=1, use_activation=True, name=None):
        layer = Conv2DTranspose(features, kernel_size, strides=strides, padding="same", name=name)(before)
        if use_activation: layer = Activation("relu")(layer)
        return layer

    def ResBlock(self, before, kernel_size=(3, 3), strides=1):
        features = int_shape(before)[-1]
        # Different from the paper scale recurrent network for deep image deblurring  which use
        # Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper which doesn't use BatchNorm because only 2 input were given
        conv1 = self.conv2d(before, features, kernel_size, strides)
        conv2 = self.conv2d(conv1, features, kernel_size, strides, False)
        add1 = add([before, conv2])
        return add1

    def EBlock(self, before, kernel_size=(3, 3), strides=1):
        # Tensorflow: channels at the end
        features = int_shape(before)[-1]
        conv1 = self.conv2d(before, features * 2, kernel_size, strides=2)
        res_block = self.ResBlock(conv1, kernel_size, strides)
        res_block = self.ResBlock(res_block, kernel_size, strides)
        res_block = self.ResBlock(res_block, kernel_size, strides)
        return res_block

    def DBlock(self, before, block_to_connect=None, kernel_size=(3, 3), strides=1):
        features = int_shape(before)[-1]
        if block_to_connect is not None:
            add1 = add([block_to_connect, before])
            res_block = self.ResBlock(add1, kernel_size, strides)
        else:
            res_block = self.ResBlock(before, kernel_size, strides)
        res_block = self.ResBlock(res_block, kernel_size, strides)
        res_block = self.ResBlock(res_block, kernel_size, strides)
        # Upsampling reducing the features
        deconv1 = self.deconv2d(res_block, features // 2, (2, 2), 2)
        return deconv1

    def InBlock(self, input, features, kernel_size=(3, 3), strides=1):
        # conv1 = self.conv2d(input, features, kernel_size, strides)
        # start with a concatblock instead of 256x256x32 conv2d layer

        input_block = self.ConcatBlock(input, features, kernel_size)
        res = self.ResBlock(input_block, kernel_size, strides)
        res = self.ResBlock(res, kernel_size, strides)
        res = self.ResBlock(res, kernel_size, strides)
        return res

    def OutBlock(self, before, output_channel, block_to_connect, kernel_size=(3, 3), strides=1, name=None):
        add1 = add([block_to_connect, before])
        res = self.ResBlock(add1, kernel_size, strides)
        res = self.ResBlock(res, kernel_size, strides)
        res = self.ResBlock(res, kernel_size, strides)
        # layer for the output image
        conv = self.deconv2d(res, output_channel, kernel_size, 1, use_activation=False, name=name)
        return conv

    def ConcatBlock(self, before, start_features, kernel_size):
        layer1 = self.conv2d(before, start_features, kernel_size)
        layer2 = self.conv2d(layer1, start_features, kernel_size)
        conc1 = concatenate([layer1, layer2])
        return conc1

    def ScaleBlock(self, input_layer, output_channel, kernel_size, prev_lstm_state=None, name=None):
        strides = 1

        # start_features = 32
        start_features = 8

        in_block = self.InBlock(input_layer, start_features, kernel_size=kernel_size, strides=strides)

        eblock1 = self.EBlock(in_block, kernel_size, strides=strides)

        eblock2 = self.EBlock(eblock1, kernel_size, strides=strides)

        prev_shape = list(int_shape(eblock2))
        new_shape = prev_shape
        features = new_shape[-1]
        new_shape.pop(0)
        new_shape = (1, *new_shape)

        lstm = Reshape(target_shape=new_shape)(eblock2)

        # BUG: can't use initial_state -> https://github.com/keras-team/keras/issues/9761#issuecomment-567915470
        lstm = ConvLSTM2D(features, kernel_size=kernel_size, padding="same",
                                return_state=False, input_shape=new_shape, data_format='channels_last',
                                kernel_constraint=max_norm(3))(lstm)
        lstm_state = None

        dblock2 = self.DBlock(lstm, kernel_size=kernel_size, strides=strides)

        dblock1 = self.DBlock(dblock2, block_to_connect=eblock1, kernel_size=kernel_size, strides=strides)

        out_block = self.OutBlock(dblock1, output_channel, block_to_connect=in_block, kernel_size=kernel_size,
                                  name=f"{name}_output")

        return out_block, lstm_state

    def _set_model(self):

        input1 = Input((*self.patch_size_reduced, 3))

        kernel_size = (5, 5)

        output1, lstm_state = self.ScaleBlock(input1, 3, kernel_size=kernel_size, name=f"scale{int_shape(input1)[1]}")
        output1_up = UpSampling2D(interpolation="bilinear")(output1)

        input2 = Input((*self.patch_size, 3))

        add1 = add([input2, output1_up])
        output2, lstm_state = self.ScaleBlock(add1, 3, kernel_size=kernel_size, name=f"scale{int_shape(input2)[1]}",
                                              prev_lstm_state=lstm_state)

        self._model = Model([input1, input2], [output1, output2])

    def mse_all_axis(self, y_true, y_pred):
        import keras.backend as K
        # the batch is avereged internally in the code
        return K.mean(K.square(y_true - y_pred), [1, 2, 3])

    def set_custom_objects(self):
        super(SRNDeblur, self).set_custom_objects()
        self.custom_objects.update({"mse_all_axis": self.mse_all_axis})

    def compile(self):
        from keras.optimizers import Adam
        from basemodel.metrics import metrics
        # Keras will make a mean of the sum of the two euclidean distance
        if self._model is None: self._set_model()
        # lr_sched = tf.keras.optimizers.schedules.PolynomialDecay(1e-4, 2000, 1e-6, 0.3)  # as in the paper
        compile_args = {"optimizer": Adam(1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                        "loss": [self.mse_all_axis, self.mse_all_axis],
                        "loss_weights": [1, 1],
                        "metrics": metrics}
        self._model.compile(**compile_args)

    def _data(self):
        from basemodel.generator.reds_generator import reds_generators

        train_gen = reds_generators("train",
                                    batch_size=self.batch_size, num_patches=self.num_patches, low_res=False,
                                    patch_size=self.patch_size)

        val_gen = reds_generators("val",
                                  batch_size=self.batch_size, num_patches=self.num_patches, low_res=False,
                                  patch_size=self.patch_size)

        test_gen = reds_generators("test",
                                   batch_size=self.batch_size, num_patches=self.num_patches, low_res=self.low_res,
                                   patch_size=self.patch_size)

        return (train_gen, val_gen, test_gen)

    def test(self):
        if self.low_res:
            self.test_low_res()
        else:
            self.test_high_res()

    def test_high_res(self):
        from random import randint
        import matplotlib.pyplot as plt
        import numpy as np

        (_, _, test_gen) = self._data()
        batch_index = randint(0, test_gen.__len__())
        batch_blur, batch_sharp = test_gen.__getitem__(batch_index)

        n = 5

        fig, axes = plt.subplots(n, 3, figsize=(10, 10), gridspec_kw={'width_ratios': [5, 5, 5]})
        fig.tight_layout()
        for i in range(n):
            blurred = batch_blur[i]

            sharp = batch_sharp[i]

            blurred_patches = self.get_patches_for_predict(blurred, self.original_size,
                                                           patch_size=self.patch_size)
            blurred_patches_resized = np.array(tf.image.resize(blurred_patches, self.patch_size_reduced))

            _, reconstructed_patches = self._model.predict([blurred_patches_resized, blurred_patches],
                                                           batch_size=self.batch_size)

            reconstructed = self.restore_image(reconstructed_patches, self.original_size)

            axes[i, 0].imshow(blurred)
            axes[i, 0].set_title("Input")
            axes[i, 1].imshow(reconstructed)
            axes[i, 1].set_title("Predicted image")
            axes[i, 2].imshow(sharp)
            axes[i, 2].set_title("True image")

        plt.savefig(self.model_dir / "test.pdf", bbox_inches="tight")
        print(f"Image save at: {str(self.model_dir / 'test.pdf')}")

    def test_low_res(self):
        from basemodel.generator.reds_generator import reds_generators
        from random import randint
        import matplotlib.pyplot as plt
        import numpy as np

        test_gen = reds_generators("test", self.batch_size, low_res=True, patch_size=(256, 256))
        # generate 5 images
        for k in range(5):
            batch_index = randint(0, test_gen.__len__())

            batch_blur, batch_sharp = test_gen.__getitem__(batch_index)
            batch_blur_resized = np.array(tf.image.resize(batch_blur, self.patch_size_reduced))

            _, batch_reconstructed = self._model.predict([batch_blur_resized, batch_blur], batch_size=self.batch_size)

            n = 5

            fig, axes = plt.subplots(n, 3, figsize=(10, 10), gridspec_kw={'width_ratios': [5, 5, 5]})
            fig.tight_layout()
            for i in range(n):
                blurred = batch_blur[i]
                sharp = batch_sharp[i]
                reconstructed = batch_reconstructed[i]

                axes[i, 0].imshow(blurred)
                axes[i, 0].set_title("Input")
                axes[i, 1].imshow(reconstructed)
                axes[i, 1].set_title("Predicted image")
                axes[i, 2].imshow(sharp)
                axes[i, 2].set_title("True image")

            plt.savefig(self.model_dir / f"test.low_res.{k}.pdf", bbox_inches="tight")
            print(f"Image save at: {str(self.model_dir / f'test.low_res.{k}.pdf')}")

    def open_test(self):
        import subprocess
        import random
        k = random.randint(1,5)
        name = "test.pdf" if not self.low_res else f"test.low_res.{k}.pdf"
        img = str(self.model_dir / name)
        subprocess.run(["explorer", img])

    def evaluate(self):
        self.evaluate_images()

    def evaluate_images(self):
        from keras.losses import mse as mean_squared_error
        import numpy as np
        from keras.backend import constant
        from datetime import datetime

        _, _, test_gen = self._data()

        iterations = 100
        # total_measurament = (iterations) * (batch_size) = 500
        metrics = {"mse": [], "psnr": [], "ssim": []}

        start = datetime.now()

        for n in range(iterations):
            print(f"\rComputing iteration {n}...", end="")
            batch_index = np.random.randint(0, test_gen.__len__())
            batch_blur, batch_sharp = test_gen.__getitem__(batch_index)

            for i in range(len(batch_sharp)):
                patches = self.get_patches_for_predict(batch_blur[i], self.original_size, self.patch_size)
                patches_resized = np.array(tf.image.resize(patches, self.patch_size_reduced))

                [_, patches_reconstructed] = self._model.predict([patches_resized, patches], batch_size=self.batch_size,
                                                                 workers=2, use_multiprocessing=True)
                # constant: transform numpy array to Tensor in order to be able to use it with tf.image.*
                reconstructed = constant(self.restore_image(patches_reconstructed, self.original_size))
                sharp = constant(batch_sharp[i])

                mse = mean_squared_error(sharp, reconstructed)
                ssim = tf.image.ssim(sharp, reconstructed, max_val=1.0)
                psnr = tf.image.psnr(sharp, reconstructed, max_val=1.0)

                metrics["mse"].append(mse)
                metrics["ssim"].append(ssim)
                metrics["psnr"].append(psnr)
        end = datetime.now()

        means = {key: np.mean(metrics[key]) for key in metrics}

        print(f"time spent: {str(end - start)}")
        print(means)

    def get_patches_for_predict(self, image, image_size, patch_size):
        import numpy as np
        patches = np.array(self.extract_patches(image, image_size, patch_size))
        return patches

    def extract_patches(self, image, image_size, patch_size):
        """
        Extract patches from an image
        :param image:
        :param image_size: (width,height,channels)
        :param patch_size: (patch_width,patch_height,channels)
        :return: list of ndarray
        """
        patches = []
        image_width, image_height = image_size[0], image_size[1]
        patch_width, patch_height = patch_size[0], patch_size[1]
        col_n_patches = image_width // patch_width + (0 if image_width % patch_width == 0 else 1)
        row_n_patches = image_height // patch_height + (0 if image_height % patch_height == 0 else 1)
        for row in range(row_n_patches):
            for col in range(col_n_patches):
                x = col * patch_width if col * patch_width + patch_width < image_width else image_width - patch_width
                y = row * patch_height if row * patch_height + patch_height < image_height else image_height - patch_height
                # Images are represented as (height,width,channels) in PIL
                patches.append(image[y:y + patch_height, x:x + patch_width])

        return patches

    def restore_image(self, patches, image_size):
        """
        Restore an image concatenating patches
        :param patches: list of patches
        :param image_size: (width,height,channels)
        :return: image
        """
        import numpy as np
        # Images are represented as (height,width,channels) in PIL
        image = np.zeros((image_size[1], image_size[0], image_size[2]))
        image_width, image_height = image_size[0], image_size[1]
        patch_size = patches[0].shape
        patch_width, patch_height = patch_size[0], patch_size[1]
        # Allow overlapping patches if // is not an integer
        col_n_patches = image_width // patch_width + (0 if image_width % patch_width == 0 else 1)
        row_n_patches = image_height // patch_height + (0 if image_height % patch_height == 0 else 1)
        for i in range(len(patches)):
            col = i % col_n_patches
            row = i // col_n_patches
            x = col * patch_width if col * patch_width + patch_width < image_width else image_width - patch_width
            y = row * patch_height if row * patch_height + patch_height < image_height else image_height - patch_height
            image[y:y + patch_height, x:x + patch_width, :] = patches[i]
        return image
