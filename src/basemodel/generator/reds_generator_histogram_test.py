import random
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, Value


def get_dirs(subset, quality, low_res=True) -> Path:
    from src.util import get_reds_dir
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


low_res = False
patch_size = (256, 256)
num_patches = 2
input2_size = patch_size
input1_size = list(map(lambda x: x // 2, input2_size))
original_size = (320, 180) if low_res else (1280, 720)

type = "train"

dir_sharp = get_dirs(type, "sharp", low_res=low_res)
dir_blur = get_dirs(type, "blur", low_res=low_res)

current = Value('i', 0)

selected_files_sharp = list(dir_sharp.glob("**/*.png"))
selected_files_blur = [str(file.absolute()).replace("sharp", "blur") for file in
                       selected_files_sharp]

selected_files_sharp_half = list(dir_sharp.glob("**/000000[5-9][0-9].png"))
selected_files_blur_half = [str(file.absolute()).replace("sharp", "blur") for file in
                            selected_files_sharp_half]

max = len(selected_files_sharp)


def get_ssim_files(selected_files_sharp, selected_files_blur, ssim_threshold=1.0):
    from PIL import Image
    from skimage.metrics import structural_similarity as compute_ssim
    import numpy as np

    name = multiprocessing.process.current_process().name
    use_threshold = ssim_threshold != 1.0
    ssim_list = list()
    if use_threshold:
        xs = random.sample(range(0, original_size[0] - input2_size[0]), 100)
        ys = random.sample(range(0, original_size[1] - input2_size[1]), 100)
    else:
        xs = random.sample(range(0, original_size[0] - input2_size[0]), num_patches)
        ys = random.sample(range(0, original_size[1] - input2_size[1]), num_patches)

    im_sharp = Image.open(selected_files_sharp)
    im_blur = Image.open(selected_files_blur)

    selected_patches = 0

    if use_threshold:
        # Get ssim threshold patches
        for x, y in zip(xs, ys):
            if selected_patches == num_patches: break
            box = (x, y, x + input2_size[0], y + input2_size[1])
            crop_sharp = np.array(im_sharp.crop(box))
            crop_blur = np.array(im_blur.crop(box))

            ssim = compute_ssim(crop_sharp, crop_blur, multichannel=True, gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
            if ssim < ssim_threshold:
                ssim_list.append(ssim)
                selected_patches += 1

    remains_patches = num_patches - selected_patches
    if remains_patches > 0:
        # Get general patches otherwise
        xs = random.sample(range(0, original_size[0] - input2_size[0]), remains_patches)
        ys = random.sample(range(0, original_size[1] - input2_size[1]), remains_patches)
        for x, y in zip(xs, ys):
            box = (x, y, x + input2_size[0], y + input2_size[1])
            crop_sharp = np.array(im_sharp.crop(box))
            crop_blur = np.array(im_blur.crop(box))

            ssim = compute_ssim(crop_sharp, crop_blur, multichannel=True, gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
            ssim_list.append(ssim)

    with current.get_lock():
        current.value += 1
    print(f"\r[{name}]red {current.value} over {max}", end="")
    return ssim_list


def get_ssims(filename, ssim_threshold, zip_iterable, iterations=5, num_thread=6):
    from functools import partial
    from pathlib import Path
    import pandas as pd
    path = Path(filename)

    if path.exists():
        return pd.read_csv(str(filename))
    else:
        get_ssim_files_wth_threshold = partial(get_ssim_files, ssim_threshold=ssim_threshold)
        outlist = list()
        files = list(zip_iterable)
        with Pool(num_thread) as pool:
            for n in range(iterations):
                print("Start iteration", n)
                result = pool.starmap(get_ssim_files_wth_threshold, files)
                outlist.extend(result)
                print(f"read {current.value} over {max}")
        df = pd.DataFrame([elem for sublist in outlist for elem in sublist], columns=["ssim"])
        df.to_csv(filename)
        return df


def plot_histograms(list_df, titles, legends, iterations=5):
    """
    :param list_df: List of DataFrame with column ssim
    :param legends: List of string to use as legend
    :param iterations: amount of iterations done
    :return:
    """
    import matplotlib.pyplot as plt
    n_sub = len(list_df) // 2
    fig, axes = plt.subplots(1, n_sub)
    for df_index in range(len(list_df)):
        index = df_index // 2
        list_df[df_index].ssim.plot.hist(ax=axes[index],alpha=0.3)
        axes[index].set_xlabel("ssim")
        axes[index].set_title(titles[index])
        axes[index].legend(legends)
    plt.suptitle(f"Histogram of ssim after {iterations} iterations")
    plt.savefig(f'histogram.pdf')
    plt.close()


def plot_kde(list_df, titles, iterations=5):
    """
    :param list_df: list of dataframe with ssim column
    :param titles: list of strings
    :param iterations: amount of iterations
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde

    kd_list = list()
    # Plot kde
    # Compute pdf
    # scott is used by df.plot.kde()
    for df in list_df:
        kd = gaussian_kde(df.ssim.values, 'scott')
        kd_list.append(kd)

    p1, p2, p3 = 0.0, 0.9, 1.25
    num_points = 300
    x_lt09 = np.linspace(p1, p2, num_points)
    x_gt09 = np.linspace(p2, p3, num_points)

    color1 = "tab:orange"
    color2 = "tab:blue"
    color3 = "tab:gray"

    fig, axes = plt.subplots(1, len(list_df), sharey='row')
    fig.suptitle(f"KDE of the training set after {iterations} iterations")
    for index in range(len(list_df)):
        axes[index].set_title(titles[index])
        list_df[index].ssim.plot.kde('scott', ax=axes[index],color=color3)
        fb1 = axes[index].fill_between(x_lt09, kd_list[index].pdf(x_lt09), color=color1, alpha=0.2)
        color1_alpha = fb1.get_fc()[0]
        fb2 = axes[index].fill_between(x_gt09, kd_list[index].pdf(x_gt09), color=color2, alpha=0.4)
        color2_alpha = fb2.get_fc()[0]
        handles, labels = axes[index].get_legend_handles_labels()
        patch1 = mpatches.Patch(color=color1_alpha, label=kd_list[index].integrate_box_1d(p1, p2))
        patch2 = mpatches.Patch(color=color2_alpha, label=kd_list[index].integrate_box_1d(p2, p3))
        handles.append(patch1)
        handles.append(patch2)

        axes[index].legend(handles=handles, loc='upper center')

    plt.savefig(f'kde.pdf')
    plt.close()


def get_histogram_patches_quality():
    iterable_full = zip(selected_files_sharp, selected_files_blur)
    iterable_half = zip(selected_files_sharp_half, selected_files_blur_half)

    num_thread = 10
    iterations = 5

    ssim_full = get_ssims("ssim.csv", 1.0, iterable_full, iterations=iterations, num_thread=num_thread)
    ssim_half = get_ssims("ssim_half.csv", 1.0, iterable_half, iterations=iterations, num_thread=num_thread)
    ssim_full_09 = get_ssims("ssim_0.9.csv", 0.9, iterable_full, iterations=iterations, num_thread=num_thread)
    ssim_half_09 = get_ssims("ssim_0.9.csv", 0.9, iterable_half, iterations=iterations, num_thread=num_thread)

    list_df = [ssim_full, ssim_full_09, ssim_half, ssim_half_09]
    titles = ["Full training set",
              "Half training set"]
    legends = ["ssim = 1.0", "ssim = 0.9"]
    plot_histograms(list_df, titles, legends)

    titles = ["FULL ssim=1.0", "FULL ssim=0.9", "HALF ssim=1.0", "HALF ssim=0.9"]
    plot_kde(list_df, titles)


"""
# compute probability and save it on a file
    with open(f"prob {ssim_threshold} ", 'w') as f:
        h = f"Type\tProb. {p1}-{p2}\tProb {p2}-{p3}"
        f.writelines([h, "\n"])
        for k in [kd, kd_half]:
            prob1 = k.integrate_box_1d(p1, p2)
            prob2 = k.integrate_box_1d(p2, p3)

            r1 = f"Full\t{prob1:.3f}\t{prob2:.3f}"
            f.writelines([r1, "\n"])
"""

if __name__ == '__main__':
    get_histogram_patches_quality()
