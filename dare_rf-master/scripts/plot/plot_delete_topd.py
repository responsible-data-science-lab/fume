"""
Plot detailed results for a single dataset using both adversaries.
"""
import os
import argparse
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
sys.path.insert(0, here + '/../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_delete_cbg import gini_dataset_dict
from plot_delete_cbg import entropy_dataset_dict


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def main(args):
    print(args)

    # get selected hyperparameters
    dataset_dict = gini_dataset_dict if args.criterion == 'gini' else entropy_dataset_dict

    # matplotlib settings
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=17)
    plt.rc('axes', titlesize=17)
    plt.rc('legend', fontsize=11)
    plt.rc('legend', title_fontsize=15)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    n_rows = len(args.subsample_size)

    # setup figure
    width = 5
    width, height = set_size(width=width * 3, fraction=1, subplots=(n_rows, 3))

    fig = plt.figure(figsize=(width, height * 0.9))
    gs = fig.add_gridspec(2, 3)

    shape_size = 10
    shape_list = ['o', '^', 's', 'd']

    adversaries = ['Random', 'Worst-of-1000']
    tol_list = ['0.1%', '0.25%', '0.5%', '1.0%']

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    # get retrain results
    if args.num_retrains:
        retrain_fp = os.path.join(args.in_dir, 'n_retrain.csv')
        retrain_y_label = 'No. retrains'
    else:
        retrain_fp = os.path.join(args.in_dir, 'retrain_cost.csv')
        retrain_y_label = 'No. samples'
    retrain_df = pd.read_csv(retrain_fp)

    for i, subsample_size in enumerate(args.subsample_size):

        metric = dataset_dict[args.dataset][0]
        n_trees = dataset_dict[args.dataset][1]
        max_depth = dataset_dict[args.dataset][2]
        k = dataset_dict[args.dataset][3]

        # filter results
        df = main_df[main_df['dataset'] == args.dataset]
        df = df[df['criterion'] == args.criterion]
        df = df[df['subsample_size'] == subsample_size]
        df = df[df['n_estimators'] == n_trees]
        df = df[df['max_depth'] == max_depth]
        df = df[df['k'] == k]

        topd0_df = df[df['topd'] == 0]

        # plot efficiency
        if i == 0:
            ax = fig.add_subplot(gs[i, 0])
            prev_efficiency_ax = ax
        else:
            ax = fig.add_subplot(gs[i, 0], sharey=prev_efficiency_ax)

        ax.set_yscale('log')
        ax.set_ylabel('({})\nSpeedup vs Naive'.format(adversaries[i], subsample_size))
        ax.errorbar(df['topd'], df['model_n_deleted'], yerr=df['model_n_deleted_std'],
                    label='R-DaRE', color='k')

        for tol, topd, shape in zip(tol_list, dataset_dict[args.dataset][4], shape_list):
            x = df['topd'].iloc[topd]
            y = df['model_n_deleted'].iloc[topd]
            ax.plot(x, y, 'k{}'.format(shape), label='tol={}'.format(tol), ms=shape_size)

        ax.axhline(topd0_df['model_n_deleted'].values[0], color='k', linestyle='--', label='G-DaRE')

        if args.dataset == 'higgs':
            ax.set_ylim(2e2, 1e6)

        if i == 0:
            # ax.legend(ncol=2, frameon=False)
            ax.set_title('Deletion Efficiency')
        elif i == n_rows - 1:
            ax.set_xlabel(r'$d_{\mathrm{rmax}}$')

        # plot utility
        if subsample_size == 1:
            ax = fig.add_subplot(gs[:, 1])
            ax.set_ylabel(r'Test error $\Delta$ (%)')
            ax.errorbar(df['topd'], df['{}_diff_mean'.format(metric)] * 100,
                        yerr=df['{}_diff_sem'.format(metric)] * 100, color='k', label='R-DaRE')

            for tol, topd, shape in zip(tol_list, dataset_dict[args.dataset][4], shape_list):
                x = df['topd'].iloc[topd]
                y = df['{}_diff_mean'.format(metric)].iloc[topd] * 100
                ax.plot(x, y, 'k{}'.format(shape), label='tol={}'.format(tol), ms=shape_size)

            ax.axhline(0, color='k', linestyle='--', label='G-DaRE')

            if i == 0:
                ax.legend(ncol=1, frameon=False)
                ax.set_title('Prediction Degradation')
            ax.set_xlabel(r'$d_{\mathrm{rmax}}$')

        # plot retrains
        if i == 0:
            ax = fig.add_subplot(gs[i, 2])
            prev_retrain_ax = ax
        else:
            ax = fig.add_subplot(gs[i, 2], sharey=prev_retrain_ax)

        ax.set_ylabel(retrain_y_label)

        for j, row in enumerate(df[1:].itertuples(index=False)):
            linestyle = '-' if j < 10 else '--'
            temp = retrain_df[retrain_df['id'] == row.id]
            retrains = temp.iloc[0].values[1:]
            depths = np.arange(retrains.shape[0])
            ax.plot(depths[:max_depth], retrains[:max_depth], linestyle=linestyle,
                    label='{}'.format(row.topd))

        temp = retrain_df[retrain_df['id'].isin(topd0_df['id'])]
        retrains = temp.iloc[0].values[1:]
        depths = np.arange(retrains.shape[0])
        ax.plot(depths[:max_depth], retrains[:max_depth], 'k--')

        if i == 0:
            ax.set_title('Retrains')
        if i == n_rows - 1:
            ax.set_xlabel('Retrain depth')
        if max_depth in [10, 20]:
            handles, labels = ax.get_legend_handles_labels()

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    os.makedirs(args.out_dir, exist_ok=True)

    # set retrain legend
    y_loc = 0.965 if max_depth == 20 else 0.800
    fig.legend(handles, labels, bbox_to_anchor=(0.995, y_loc), ncol=1,
               title=r'$d_{\mathrm{rmax}}$')

    fig.tight_layout(rect=[0, 0, 0.935, 1])
    fp = os.path.join(args.out_dir, '{}.pdf'.format(args.dataset))
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='credit_card', help='dataset to use for plotting.')
    parser.add_argument('--in_dir', type=str, default='output/delete/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/delete_topd/', help='output directory.')
    parser.add_argument('--criterion', type=str, default='gini', help='split criterion.')
    parser.add_argument('--subsample_size', type=int, nargs='+', default=[1, 1000], help='adversary strength.')
    parser.add_argument('--num_retrains', action='store_true', default=False, help='plot no. retrains.')
    args = parser.parse_args()
    main(args)
