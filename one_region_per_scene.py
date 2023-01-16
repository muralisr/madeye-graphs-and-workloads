
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from matplotlib.lines import Line2D
my_two_colors = ["#cccccc", "#525252"] # shades of black
my_three_colors = ["#f0f0f0","#bdbdbd","#636363"]
my_four_colors = ["#f7f7f7", "#cccccc", "#969696", "#525252"] # shades of black
my_four_colors_with_darker_start = ["#cccccc", "#969696", "#636363", "#252525"]
import json
BUCKETS = 1000


def round3(x):
    return round(x, 2)


def create_pct_diff_plot_2(
    series,
    labels,
    xlabel="% Improvement",
    ylabel="CDF",
    legend_size=12,
    save_path=None,
    location="lower right",
    log_scale=False,
    colors=None,
    curve_line_width=2,
    fig_width=5,
    fig_height=2.3,
    x_min=None,
    x_max=None,
    axes_font_size=12,
    tick_label_size=12,
    linestyles=None,
    hide_legend=False,
    rc_params_input=None,
    x_ticks=None,
):
    # colors = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0", "#f0027f", "#bf5b17", "#666666"]
    # colors = [
    #     "#fdc086",
    #     "#beaed4",
    #     "#386cb0",
    #     "#f0027f",
    #     "#bf5b17",
    #     "#666666",
    #     "#fdc086",
    #     "#beaed4",
    #     "#386cb0",
    #     "#f0027f",
    #     "#bf5b17",

    # ]
    if colors is None:
        colors = [
            "#fdc086",
            "#beaed4",
            "#386cb0",
            "#f0027f",
            "#bf5b17",
            "#666666",
            "#fdc086",
            "#beaed4",
            "#386cb0",
            "#f0027f",
            "#bf5b17",
            "#fb9a99",
            "#a6cee3"
        ]
    

    
    if rc_params_input is not None:
        matplotlib.rcParams.update(rc_params_input)
    

    if linestyles is None:
        linestyles = linestyles = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"] 
        # ["-", "--", ":", "-."]
    # linestyles = ["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    percentiles = {}
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=1000)
    for i, (serie, label, color, linestyle) in enumerate(
        zip(series, labels, colors, linestyles)
    ):
        plt.hist(
            serie,
            BUCKETS,
            color=color,
            linestyle=linestyle,
            density=True,
            histtype="step",
            cumulative=True,
            label=label,
            linewidth=curve_line_width,
        )
        print(
            label,
            tuple(
                map(
                    round3,
                    (
                        np.percentile(serie, 5),
                        np.percentile(serie, 25),
                        np.percentile(serie, 50),
                        np.percentile(serie, 75),
                        np.percentile(serie, 95),
                    ),
                )
            ),
        )
        percentiles[label] = [
            np.percentile(serie, 5),
            np.percentile(serie, 25),
            np.percentile(serie, 50),
            np.percentile(serie, 75),
            np.percentile(serie, 80),
            np.percentile(serie, 85),
            np.percentile(serie, 90),
            np.percentile(serie, 95),
        ]
    if hide_legend is False:
        # custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
        #         Line2D([0], [0], color=colors[1], lw=2)]
        custom_lines = [Line2D([0], [0], color=colors[x], lw=2, linestyle=linestyles[x]) for x in range(len(colors))]
        ax.legend(
            custom_lines, labels,
            loc=location,
            prop={"size": legend_size},
            handlelength=0.7,
            handletextpad=0.5,
        )
    if axes_font_size is None:
        axes_font_size = legend_size
    if tick_label_size is None:
        tick_label_size = legend_size
    plt.tick_params(axis="both", which="major", labelsize=tick_label_size)
    plt.ylabel(ylabel, fontsize=axes_font_size)
    plt.xlabel(xlabel, fontsize=axes_font_size)
    if log_scale:
        plt.xscale("log")
    plt.tight_layout()
    plt.ylim(0, 1)
    if x_ticks is not None:
        print(f"setting xtics")
        ax.set_xticks(x_ticks)
    ax.set_yticks([0,0.25,0.5,0.75,1])
    if not log_scale:
        # print(f'series is {series}')
        if x_min is None:
            plt.xlim(min(map(min, series)), max(map(max, series)))
        else:
            plt.xlim(x_min, x_max)

    fix_hist_step_vertical_line_at_end(ax)
    # matplotlib.rcParams.update({
    #     "figure.subplot.left":0.117,
    #     "figure.subplot.bottom":0.23, 
    #     "figure.subplot.right":0.96, 
    #     "figure.subplot.top":0.962, 
    #     "figure.subplot.wspace":0.2, 
    #     "figure.subplot.hspace":0.2
    # })
    if save_path:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    else:
        plt.show()
    return percentiles

dataset=[
  {
    "1": 1.0,
    "2": 0.9267633201571027,
    "5": 0.8576520917194497,
    "10": 0.8097123803082358,
    "1000": 0.4711749102940815
  },
  {
    "1": 0.6666666666666666,
    "2": 0.5850460386174672,
    "5": 0.5030264378478665,
    "10": 0.48927841338555617,
    "1000": 0.16135118188689612
  },
  {
    "1": 1.0,
    "2": 0.9437306283460127,
    "5": 0.9191673710904479,
    "10": 0.9120785197708272,
    "1000": 0.9661665257819103
  },
  {
    "1": 1.0,
    "2": 0.9135531135531133,
    "5": 0.8641069248212104,
    "10": 0.8129235843521557,
    "1000": 0.1586538461538462
  },
  {
    "1": 1.0,
    "2": 0.8902735108380269,
    "5": 0.843671934397741,
    "10": 0.789125510093252,
    "1000": 0.5882112212757372
  },
  {
    "1": 1.0,
    "2": 0.8649107142857143,
    "5": 0.6930654761904761,
    "10": 0.665019841269841,
    "1000": 0.39007605820105823
  }
]

to_plot = {
    "N=1 sec": [item ["1"] for item in dataset],
    "N=2 sec": [item ["2"] for item in dataset],
    "N=5 sec": [item ["5"] for item in dataset],
    "N=1000 sec": [item ["1000"] for item in dataset],
}

create_pct_diff_plot_2(
            list(to_plot.values()),
            to_plot.keys(),
            legend_size=20,
            save_path="effect_of_retricting_frequency_of_switches_to_best_orientation-one_region_per_scene.pdf",
            xlabel="Accuracy (fraction) we can achieve when we restrict oracle from switching to the best orientation every N seconds.\nN=1 means the oracle switches to the best orientation at every second.\nN=1000 means the oracle switches to the best orientation at t=0 and stays there (as video length is less than 1000 seconds).\nEntire scene is one region, one score per video, 5 videos.",
            ylabel="CDF",
            location="best",
            log_scale=False,
            curve_line_width=2,
            fig_width=15,
            fig_height=7.5,
            axes_font_size=14,
    )