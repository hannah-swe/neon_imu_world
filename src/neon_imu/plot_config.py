import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def setup_plot_style():
    # 1) Basis Theme
    sns.set_theme(
        context="talk",
        style="ticks",
    )

    # 2) Custom crest palette with extremes + middle
    cmap = sns.color_palette("crest", as_cmap=True)

    positions = [0.1, 0.5, 1.0]  # min, middle, max
    custom_palette = [cmap(p) for p in positions]

    sns.set_palette(custom_palette)

    # 3) Global figure settings
    plt.rcParams["figure.figsize"] = (8, 7)
    plt.rcParams["figure.autolayout"] = True

    # 4) Always despine
    original_show = plt.show

    def show_with_despine(*args, **kwargs):
        sns.despine()
        original_show(*args, **kwargs)

    plt.show = show_with_despine