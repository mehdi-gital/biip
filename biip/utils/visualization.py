import numpy as np
import matplotlib.pyplot as plt


def plot_true_pred_cylinder(
        title,
        grid_size_i,
        grid_size_j,
        dataset,
        f_hat,
        save_path,
        fix_scale=False
):
    figure_title = title
    fig, axs = plt.subplots(grid_size_i, grid_size_j + 2)

    fig.set_size_inches(12, 9)
    fig.suptitle(figure_title, fontsize=16)

    num_timesteps = dataset['num_timesteps']

    f_hat = f_hat.squeeze().numpy()
    f_hat = f_hat.reshape(num_timesteps, grid_size_i, grid_size_j)

    f_interior = dataset['f_interior'].numpy().reshape(num_timesteps, grid_size_i, grid_size_j)
    f_boundary = dataset['f_boundary'].squeeze().numpy().reshape(num_timesteps, grid_size_i, 2)

    # plotting the boundary
    for i in range(dataset['num_nodes_boundary']//2):
        axs[i, 0].tick_params(axis='both', which='major', labelsize=3)
        axs[i, 0].tick_params(axis='both', which='minor', labelsize=3)
        axs[i, 0].plot([t for t in range(num_timesteps)], f_boundary[:, i, 0],
                       color='limegreen', linestyle='dotted', linewidth=4, label='Ground truth', zorder=4)

        axs[i, 7].tick_params(axis='both', which='major', labelsize=3)
        axs[i, 7].tick_params(axis='both', which='minor', labelsize=3)
        axs[i, 7].plot([t for t in range(num_timesteps)], f_boundary[:, i, 1],
                       color='limegreen', linestyle='dotted', linewidth=4, label='Ground truth', zorder=4)
        axs[i, 0].set_facecolor('gainsboro')
        axs[i, 7].set_facecolor('gainsboro')
        if fix_scale:
            axs[i, 0].set_ylim([0, dataset['f_boundary'].max().item() * 1.1])
            axs[i, 7].set_ylim([0, dataset['f_boundary'].max().item() * 1.1])

    # interior
    for i in range(grid_size_i):
        for j in range(1, grid_size_j + 1):
            axs[i, j].tick_params(axis='both', which='major', labelsize=3)
            axs[i, j].tick_params(axis='both', which='minor', labelsize=3)
            axs[i, j].plot([t for t in range(num_timesteps)], f_interior[:, i, j - 1],
                           color='limegreen', linestyle='dotted', linewidth=4, label='Ground truth', zorder=5)

            axs[i, j].plot([t for t in range(num_timesteps)], f_hat[:, i, j - 1],
                           color='b', label='Ours', zorder=4)
            axs[i, j].set_facecolor('whitesmoke')
            if fix_scale:
                axs[i, j].set_ylim([0, dataset['f_interior'].max().item() * 1.1])

    handles, labels = axs[i, j].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.savefig(save_path, format='png', dpi=250)
    plt.clf()
    plt.cla()
    plt.close('all')


def plot_loss_curve(losses, title, ylabel, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(title)

    # # smoothing
    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size
    losses = np.convolve(np.log(losses), kernel, mode='same')

    e = [i for i in range(len(losses) - 10)]
    ax.plot(e, losses[5:-5], color='b', label='model', zorder=1)

    ax.set_facecolor('whitesmoke')
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(save_path, format='png', dpi=250)
