def plot_history(save_dir, history):
    import matplotlib.pyplot as plt
    plt.set_cmap("Pastel1")

    keys = [key for key in history.history.keys() if "val" not in key]

    for key in keys:
        plt.figure()
        x = range(1, len(history.history[key]) + 1)
        plt.xticks(x)

        y = history.history[key]
        plt.plot(x, y, label=key)

        label = f"val_{key}"
        y = history.history[label]

        plt.plot(x, y, label=label)

        plt.legend()

        plt.xlabel("Number of epochs")

        plt.savefig(save_dir / f"training.{key}.pdf")


def plot_csv(path, what, title=None, best_epoch=None):
    """
    Plot content of a csv file
    :param path: path of the file .csv
    :param what: list of string. What to print: ['loss','ssim']. Automatically is added val as prefix
    :return:
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sbn
    sbn.set(style="white", palette="pastel",)
    from pathlib import Path
    path = Path(path)
    folder = path.parent
    file_to_save = folder / f"plot_history_{title}.pdf"

    df = pd.read_csv(path, index_col="epoch")
    df.index = df.index + 1

    # shift range epoch column
    if best_epoch is None and "loss" in what:
        best_epoch = df[f"val_loss"].idxmin()

    n_row = len(what)

    fig, axes = plt.subplots(n_row, 1, sharex='col')

    for index in range(n_row):
        what_to_plot = [what[index], f"val_{what[index]}"]
        df.plot.line(y=what_to_plot, ax=axes[index], use_index=True)

        # best value taken from val_
        best_value = df.loc[best_epoch, what_to_plot[index]]
        # show the best value with lines
        # plot horizontal line
        x = list(range(1, best_epoch))
        y = [best_value] * len(x)
        axes[index].plot(x, y, '--r')
        # plot vertical
        y = np.linspace(0, best_value, 100)
        x = [best_epoch] * len(y)
        axes[index].plot(x, y, '--r')
        # add xtick of best epoch
        if index == 0:
            xticks = [x for x in axes[index].get_xticks()]

            # remove two nearest number from x-axis
            for k in range(1, 3):
                if best_epoch + k in xticks:
                    xticks.remove(best_epoch + k)
                if best_epoch - k in xticks:
                    xticks.remove(best_epoch - k)
            if best_epoch not in xticks:
                xticks.append(best_epoch)

            axes[index].set_xticks(xticks)
        # add ytick of best value
        axes[index].set_yticks(np.append([tick for tick in axes[index].get_yticks() if tick > 0], best_value))

    plt.suptitle(title)
    plt.savefig(file_to_save, bbox_inches="tight")
    print(f"Plot saved at: {file_to_save}")
