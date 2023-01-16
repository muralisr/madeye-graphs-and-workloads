
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

def get_list_from_json_file(input_file):
    with open(input_file, "r") as f:
        return list(json.loads(f.read()).values())

def round3(x):
    return round(x, 2)

def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [
        poly
        for poly in ax.get_children()
        if isinstance(poly, matplotlib.patches.Polygon)
    ]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

def crop_pdf_file(input_file, percentage=5, ignore_bottom=False, right_percent=None):
    if right_percent is None:
        right_percent = percentage
    if ignore_bottom is False:
        os.system(f"python /disk/Code/projects/floo_graphs/crop_tool/pdfCropMargins/bin/pdfCropMargins.py -p4 {percentage} {percentage} {right_percent} {percentage} {input_file} -o {input_file.replace('.pdf', '.cropped.pdf')}")
    else:
        os.system(f"python /disk/Code/projects/floo_graphs/crop_tool/pdfCropMargins/bin/pdfCropMargins.py -p4 {percentage} 100 {percentage} {percentage} {input_file} -o {input_file.replace('.pdf', '.cropped.pdf')}")


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


def do_clustered_bar_plotting(
    list_of_time_delta_names,
    list_of_medians,
    list_of_25ths,
    list_of_75ths,
    list_of_network_names,
    output_file_name,
    y_axis_name=None,
    x_axis_name=None,
    label_size=17,
    tick_label_size=None,
    fig_width=7,
    fig_height=6,
    colors=None,
    legend_size=None,
    use_patterns=True,
    legend_loc=None,
    legend_handle_size=None,
    legend_handletextpad=None,
    rc_params_input = None,
    legend_num_cols=None,
    y_is_log=False,
    fit_legend_in_top_center=False,
    fit_legend_in_top_center_for_runtime=False,
    fit_legend_in_top_center_for_tango=False
):

    if rc_params_input is None:

        matplotlib.rcParams.update(
        {
            "figure.subplot.left":0.125,
            "figure.subplot.bottom":0.195, 
            "figure.subplot.right":0.9, 
            "figure.subplot.top":1, 
            "figure.subplot.wspace":0.2, 
            "figure.subplot.hspace":0.2
        })
    else:
        matplotlib.rcParams.update(rc_params_input)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    # fig.SubplotParams(left=0.125, bottom=0.179, right=0.9, top=1, wspace=0.2, hspace=0.2)
    width = 0.15
    list_of_bars = []
    ind = np.arange(len(list_of_time_delta_names))
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
    
    linestyles = ["-.", "--"]
    hatches=["*", "\\", "\//", "x", "o", "O", ".", "*"]

    for i in range(len(list_of_network_names)):
        if list_of_25ths is None:
            p = ax.bar(
                ind + (width * i),
                list_of_medians[i],
                width,
                color=colors[i],
                edgecolor="black",
                linewidth=0.4
            )
        else:
            p = ax.bar(
                ind + (width * i),
                list_of_medians[i],
                width,
                yerr=[list_of_25ths[i], list_of_75ths[i]],
                color=colors[i],
                edgecolor="black",
                linewidth=0.4,
                error_kw={'capsize':3, 'elinewidth': 0.6}
            )
        if i > 0 and use_patterns:
            for p1 in p:
                p1.set_hatch(hatches[i])
        list_of_bars.append(p)
    # ax.set_title(
    #     "median % optimization achieved per interaction. error bars represent 25, 75th percentile"
    # )
    
    if y_axis_name is not None:
        # ax.set_ylabel(y_axis_name, {'fontsize': 18})
        plt.ylabel(y_axis_name, fontsize=label_size)
    if x_axis_name is not None:
        # ax.set_ylabel(x_axis_name, {'fontsize': 18})
        plt.xlabel(x_axis_name, fontsize=label_size)
    if tick_label_size is None:
        tick_label_size = label_size
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(list_of_time_delta_names)
    if legend_size is None:
        legend_size = label_size
    if legend_loc is None:
        legend_loc = "upper left"
    if legend_handle_size is None:
        legend_handle_size = 3
    if legend_handletextpad is None:
        legend_handletextpad = 0.5
    if legend_num_cols is None:
        legend_num_cols = 1
    if fit_legend_in_top_center:
        legend_bars = [x[0] for x in list_of_bars]
        ax.legend(legend_bars, list_of_network_names, handlelength=0.515,
        handletextpad=legend_handletextpad, borderpad=-0.5, fontsize=legend_size, ncol=legend_num_cols,  mode='expand', bbox_to_anchor=(0, 0.9, 1, 0.2), fancybox=False, edgecolor='white')
    elif fit_legend_in_top_center_for_runtime:
        ax.legend([x[0] for x in list_of_bars], list_of_network_names, handlelength=0.715,
        handletextpad=legend_handletextpad, borderpad=0, fontsize=legend_size, ncol=legend_num_cols,  mode='expand', bbox_to_anchor=(0, 0.98, 1, 0.2), fancybox=False, edgecolor='white')
    elif fit_legend_in_top_center_for_tango:
        legend_bars = [x[0] for x in list_of_bars]
        legend_bars.append("_None")
        ax.legend(legend_bars, list_of_network_names, handlelength=0.515,
        handletextpad=legend_handletextpad, borderpad=-0.5, fontsize=legend_size, ncol=legend_num_cols,   bbox_to_anchor=(0, 0.9, 1, 0.2), fancybox=False, edgecolor='white')
    else:
        ax.legend([x[0] for x in list_of_bars], list_of_network_names, loc=legend_loc, handlelength=legend_handle_size,
        handletextpad=legend_handletextpad, borderpad=0.2, fontsize=legend_size, ncol=legend_num_cols)
    if y_is_log:
        plt.yscale("log")
    # plt.show()
    if output_file_name is None:
        plt.show()
    else:
        plt.savefig(output_file_name, bbox_inches='tight')


def do_scatter_plot(x_axis, y_axis, output_file_name, x_label = None, y_label = None, plt_title = "", axis_label_size = 12, fig_width=7, fig_height=6, color=None, tick_label_size=None):
    matplotlib.rcParams.update(
    {
        "figure.subplot.left":0.125,
        "figure.subplot.bottom":0.179, 
        "figure.subplot.right":0.9, 
        "figure.subplot.top":1, 
        "figure.subplot.wspace":0.2, 
        "figure.subplot.hspace":0.2
    })
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    index = 0
    if color is None:
        color = "#000000"
    while index < len(x_axis):
        ax.scatter(x_axis[index], y_axis[index], color=color, s=14)
        index += 1 
    if tick_label_size is None:
        tick_label_size = axis_label_size
    plt.tick_params(axis="both", which="major", labelsize=tick_label_size)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=axis_label_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=axis_label_size)
    ax.set_title(
        plt_title
    )
    plt.savefig(output_file_name, bbox_inches='tight')


def do_bar_chart_plotting(
    list_of_time_delta_names=None,
    list_of_medians=None,
    list_of_25ths=None,
    list_of_75ths=None,
    output_file_name=None,
    y_axis_name=None,
    x_axis_name=None,
    label_size=12,
    tick_label_size=12,
    fig_width=7,
    fig_height=6,
    colors=None,
    legend_size=12,
    hide_legend=False,
    use_patterns=True,
    legend_num_cols=1,
    list_of_legend_names=None,
    legend_loc="upper center",
    x_axis_size=None,
):

    # matplotlib.rcParams.update(
    # {
    #     "figure.subplot.left":0.125,
    #     "figure.subplot.bottom":0.195, 
    #     "figure.subplot.right":0.9, 
    #     "figure.subplot.top":1, 
    #     "figure.subplot.wspace":0.2, 
    #     "figure.subplot.hspace":0.2
    # })
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
    
    linestyles = ["-.", "--"]
    if use_patterns:
        hatches=["", "\\", "/", "\\/", "--", "o", "O", ".", "*"]
    else:
        hatches=["", "", "", "", "", "", "", ""]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    location_of_bars = [x/4 for x in np.arange(len(list_of_time_delta_names))]

    p = ax.bar(location_of_bars, list_of_medians, yerr=[list_of_25ths, list_of_75ths], color=colors, edgecolor="black", linewidth=0.4, width=0.15, align='center', error_kw={'capsize':5})
    for bar, pattern in zip (p, hatches):
        bar.set_hatch(pattern)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
    ax.set_xticks(location_of_bars)
    ax.set_xticklabels(list_of_time_delta_names)
    # ax.yaxis.grid(True)

    
    if y_axis_name is not None:
        # ax.set_ylabel(y_axis_name, {'fontsize': 18})
        plt.ylabel(y_axis_name, fontsize=label_size)
    if x_axis_name is not None:
        # ax.set_xlabel(x_axis_name, {'fontsize': 18})
        print(f"setting x axis name to {x_axis_name}")
        if x_axis_size is None:
            x_axis_size = label_size
        plt.xlabel(x_axis_name, fontsize=x_axis_size)
    if tick_label_size is None:
        tick_label_size = label_size
    
    if list_of_legend_names is None:
        list_of_legend_names = list_of_time_delta_names
    if legend_size is None:
        legend_size = label_size
    if not hide_legend:
        ax.legend(p, list_of_legend_names, loc=legend_loc, handlelength=1,
            handletextpad=0.5, borderpad=0.1, fontsize=legend_size, ncol=legend_num_cols)
  
    # plt.show()
    plt.savefig(output_file_name,bbox_inches='tight')

def do_dots_plotting(x_axis, y_axis, output_file_name, x_label = None, y_label = None, plt_title = "", axis_label_size = 12, fig_width=7, fig_height=6, color=None, tick_label_size=None):
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    index = 0
    if color is None:
        color = my_four_colors
    for xe, ye in zip(x_axis, y_axis):
        plt.scatter([xe] * len(ye), ye)
    if tick_label_size is None:
        tick_label_size = axis_label_size
    plt.tick_params(axis="both", which="major", labelsize=tick_label_size)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=axis_label_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=axis_label_size)
    ax.set_title(
        plt_title
    )
    plt.savefig(output_file_name, bbox_inches='tight')


    
def do_dots_plotting_aish(x_axis, y_axis1,y_axis2, y_axis1_label, y_axis2_label, output_file_name, x_label = None, y_label = None, plt_title = "", axis_label_size = 12, fig_width=5, fig_height=2.1, color=None, tick_label_size=None, x_ticks=None, legend_loc="upper left", y_ticks=None):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    index = 0
    if color is None:
        color = my_two_colors
    # for xe, ye in zip(x_axis, y_axis):
    #     plt.scatter([xe] * len(ye), ye)
    if y_axis1 is not None:
        plt.scatter(x_axis,y_axis1, marker='x',label=y_axis1_label,color=color[-1])
    if y_axis2 is not None:
        plt.scatter(x_axis,y_axis2, marker='o',label=y_axis2_label,color=color[-2])
    if tick_label_size is None:
        tick_label_size = axis_label_size
    plt.tick_params(axis="both", which="major", labelsize=tick_label_size)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=axis_label_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=axis_label_size)
    ax.set_title(
        plt_title
    )
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if y_axis1 is not None and y_axis2 is not None:
        ax.legend(loc=legend_loc,
        prop={"size":axis_label_size},
            handlelength=0.3,
            handletextpad=0.3)
    plt.savefig(output_file_name, bbox_inches='tight')

def do_once_always_dot_plot(list_of_fractions, output_file_path):
    frac_to_num_occurence = {}
    to_plot_x = []
    to_plot_y = []
    for frac in list_of_fractions:
        y_axis = 1
        if frac not in frac_to_num_occurence:
            frac_to_num_occurence[frac] = y_axis
        else:
            frac_to_num_occurence[frac] += 1
            y_axis = frac_to_num_occurence[frac]
        to_plot_y.append(y_axis)
        to_plot_x.append(frac)
    x_values = []
    y_values = []
    for k, v in frac_to_num_occurence.items():
        x_values.append(k)
        y_values.append(v)
    plt.bar(x_values, y_values)
    print(f"saving to output")
    # x_values = x_values[0:10]
    # y_values = y_values[0:10]
    plt.savefig(output_file_path, bbox_inches='tight')


def plot_histogram(datapoints, num_bins, output_file_name):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    weights = np.ones_like(datapoints) / len(datapoints)
    ax.hist(datapoints, bins=num_bins, weights=weights)
    ax.set_xlabel("percentage of invocations that were memoizable")
    ax.set_ylabel("fraction of functions")
    plt.savefig(output_file_name)