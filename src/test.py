# # def extract_patches(image, image_size=(320, 180), patch_size=(128, 128)):
# #     # WIDTH and HEIGHT are swapped inside IMAGE
# #     patches = []
# #     image_width, image_height = image_size[0], image_size[1]
# #     patch_width, patch_height = patch_size[0], patch_size[1]
# #     col_n_patches = image_width // patch_width + (0 if image_width % patch_width == 0 else 1)
# #     row_n_patches = image_height // patch_height + (0 if image_height % patch_height == 0 else 1)
# #     for row in range(row_n_patches):
# #         for col in range(col_n_patches):
# #             y = col * patch_width if col * patch_width + patch_width < image_width else image_width - patch_width
# #             x = row * patch_height if row * patch_height + patch_height < image_height else image_height - patch_height
# #             patches.append(image[x:x + patch_width, y:y + patch_height])
# #
# #     return patches
# #
# #
# # def restore_image(patches, image_size=(180, 320, 3)):
# #     image = np.zeros(image_size)
# #     image_width, image_height = image_size[1], image_size[0]
# #     patch_size = patches[0].shape
# #     patch_width, patch_height = patch_size[0], patch_size[1]
# #     col_n_patches = image_width // patch_width + (0 if image_width % patch_width == 0 else 1)
# #     row_n_patches = image_height // patch_height + (0 if image_height % patch_height == 0 else 1)
# #     for i in range(len(patches)):
# #         col = i % col_n_patches
# #         row = i // col_n_patches
# #         y = col * patch_width if col * patch_width + patch_width < image_width else image_width - patch_width
# #         x = row * patch_height if row * patch_height + patch_height < image_height else image_height - patch_height
# #         image[x:x + patch_width, y:y + patch_height, :] = patches[i]
# #     return image
# #
#
# import numpy as np
# from random import randint
# import matplotlib.pyplot as plt
# from src.basemodel.generator.reds_generator import reds_generators
#
# # test_gen = reds_generators_bicubic("test")
# # batch_index = randint(0, test_gen.__len__())
# # [_, batch_blur], [_, batch_sharp] = test_gen.__getitem__(batch_index)
# #
# # n = 1
# #
# # for i in range(n):
# #     blurred = batch_blur[i]
# #     sharp = batch_sharp[i]
# #     patches = extract_patches(blurred)
# #
# #     fig, axes = plt.subplots(4, 3)
# #
# #     for i in range(len(patches)):
# #         axes[1 + i // 3, i % 3].imshow(patches[i])
# #     plt.show()
# #
# #     restored = restore_image(patches)
# #     plt.imshow(restored)
# #     plt.show()
#
#
# # train_gen = reds_generators("train", low_res=False, batch_size=10)
# #
# # [batch_blur_resized, batch_blur], [batch_sharp_resized, batch_sharp] = train_gen.__getitem__(1)
# #
# # for i in range(len(batch_blur)):
# #
# #     plt.subplot(211)
# #     plt.imshow(batch_blur[i])
# #     plt.subplot(212)
# #     plt.imshow(batch_sharp[i])
# #     plt.show()
# 
# # from src.basemodel.generator.reds_generator import reds_generators
# # reds_generators("val",low_res=False,patch_size=(256,256)).test_get_patches_images()
#
# from src.basemodel.generator.reds_generator import reds_generators
# reds_generators("train",low_res=False,patch_size=(256,256),num_patches=5).get_histogram_patches_quality()

from src.models.reds.SRNDeblur import SRNDeblur as SRNDeblur_reds

model = SRNDeblur_reds(low_res_test=False, last_epoch=134)

model.summary()