import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import Sequence

from src.util import get_reds_dir


def get_dirs(subset, quality, low_res=True) -> Path:
    """
    :param low_res:
    :param subset: train,val,test
    :param quality: blur, sharp
    :return:
    """
    out_path = get_reds_dir() / f"{subset}"
    if low_res:
        out_path = out_path / f"{subset}_{quality}_bicubic" / "X4"
    else:
        out_path = out_path / f"{subset}_{quality}"
    return out_path


class reds_generators(Sequence):
    def __init__(self, type, batch_size=32, num_patches=1,
                 low_res=True, patch_size=(256, 256)):

        self.low_res = low_res

        self.ssim_threshold = 0.90

        self.input2_size = patch_size
        self.input1_size = list(map(lambda x: x // 2, self.input2_size))

        self.original_size = (320, 180) if low_res else (1280, 720)

        # If low_res the image is resized to (256,256) with 0-padding thus whole the image is a patch
        self.num_patches = 1 if low_res else num_patches

        self.batch_size = batch_size

        self.type = type

        self.dir_sharp = get_dirs(type, "sharp", low_res=low_res)
        self.dir_blur = get_dirs(type, "blur", low_res=low_res)

        self.files_sharp = list(self.dir_sharp.glob("**/000000[5-9][0-9].png"))

    def __len__(self):
        return len(self.files_sharp) * self.num_patches // self.batch_size

    # Function used for plotting patches with psnr and ssim
    def test_get_patches_images(self):
        selected_files_sharp = random.choices(self.files_sharp, k=self.batch_size // self.num_patches)
        selected_files_blur = [str(file.absolute()).replace("sharp", "blur") for file in
                               selected_files_sharp]
        # # Get patches
        # xs = random.sample(range(0, self.original_size[0] - self.input2_size[0]), self.num_patches)
        # ys = random.sample(range(0, self.original_size[1] - self.input2_size[1]), self.num_patches)

        # batch_sharp = (self.get_patches_images(selected_files_sharp, xs, ys))
        # batch_blur = (self.get_patches_images(selected_files_blur, xs, ys))
        batch_sharp, batch_blur = self.get_all_patches(selected_files_sharp, selected_files_blur,
                                                       ssim_threshold=self.ssim_threshold)

        import matplotlib.pyplot as plt
        for bs, bb in zip(batch_sharp, batch_blur):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(f"psnr: {tf.image.psnr(bs, bb, max_val=1.)} \n ssim: {tf.image.ssim(bs, bb, max_val=1.)}")
            ax1.imshow(bs)
            ax2.imshow(bb)
            plt.show()

    def get_all_patches(self, images_sharp, images_blur, ssim_threshold):

        sharp_list = list()
        blur_list = list()

        for s, b in zip(images_sharp, images_blur):
            selected_patches = 0

            sharp = img_to_array(load_img(s)) / 255.
            blur = img_to_array(load_img(b)) / 255.

            # Get a consistent number of random x,y patch origins
            xs = random.sample(range(0, self.original_size[0] - self.input2_size[0]), 100)
            ys = random.sample(range(0, self.original_size[1] - self.input2_size[1]), 100)
            xy = zip(xs, ys)
            # Iterate over them until self.num_patches is reached
            for x, y in xy:
                if selected_patches == self.num_patches:
                    break
                else:
                    sharp_patch = tf.image.crop_to_bounding_box(sharp, y, x, self.input2_size[1], self.input2_size[0])
                    blur_patch = tf.image.crop_to_bounding_box(blur, y, x, self.input2_size[1], self.input2_size[0])
                    ssim = tf.image.ssim(sharp_patch, blur_patch, max_val=1.)
                    if selected_patches < self.num_patches and ssim < ssim_threshold:
                        sharp_list.append(sharp_patch)
                        blur_list.append(blur_patch)
                        selected_patches += 1
            # Fallback to general location if patches with ssim < ssim_threshould are not found
            if selected_patches != self.num_patches:
                amount = self.num_patches - selected_patches
                xs = random.sample(range(0, self.original_size[0] - self.input2_size[0]), amount)
                ys = random.sample(range(0, self.original_size[1] - self.input2_size[1]), amount)
                for x, y in zip(xs, ys):
                    sharp_patch = tf.image.crop_to_bounding_box(sharp, y, x, self.input2_size[1], self.input2_size[0])
                    blur_patch = tf.image.crop_to_bounding_box(blur, y, x, self.input2_size[1], self.input2_size[0])
                    sharp_list.append(sharp_patch)
                    blur_list.append(blur_patch)
        return sharp_list, blur_list

    def load_low_res_images(self, sharps, blurs):
        out_sharps, out_blurs = list(),list()
        for s, b in zip(sharps, blurs):
            sharp = tf.image.resize_with_pad(img_to_array(load_img(s)) / 255., self.input2_size[1], self.input2_size[0])
            blur = tf.image.resize_with_pad(img_to_array(load_img(b)) / 255., self.input2_size[1], self.input2_size[0])
            out_sharps.append(sharp)
            out_blurs.append(blur)
        return np.array(out_sharps), np.array(out_blurs)

    def __getitem__(self, item):

        if self.type in ["train", "val"]:
            selected_files_sharp = random.choices(self.files_sharp, k=self.batch_size // self.num_patches)
            selected_files_blur = [str(file.absolute()).replace("sharp", "blur") for file in
                                   selected_files_sharp]
            if not self.low_res:
                batch_sharp, batch_blur = self.get_all_patches(selected_files_sharp, selected_files_blur,
                                                               ssim_threshold=self.ssim_threshold)
            else:
                batch_sharp, batch_blur = self.load_low_res_images(selected_files_sharp, selected_files_blur)

            batch_sharp_resized = np.array(tf.image.resize(batch_sharp, self.input1_size))
            batch_blur_resized = np.array(tf.image.resize(batch_blur, self.input1_size))

            return [batch_blur_resized, np.array(batch_blur)], [batch_sharp_resized, np.array(batch_sharp)]

        elif self.type == "test":
            selected_files_sharp = random.choices(self.files_sharp, k=self.batch_size)
            selected_files_blur = [str(file.absolute()).replace("sharp", "blur") for file in
                                   selected_files_sharp]
            # returns normal images
            if not self.low_res:
                batch_sharp = np.array(
                    [img_to_array(load_img(image_file)) / 255. for image_file in selected_files_sharp])
                batch_blur = np.array(
                    [img_to_array(load_img(image_file)) / 255. for image_file in selected_files_blur])
            else:
                batch_sharp, batch_blur = self.load_low_res_images(selected_files_sharp, selected_files_blur)

            return batch_blur, batch_sharp
        else:
            raise RuntimeError("argument << type >> is wrong")
