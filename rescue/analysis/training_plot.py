import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches


def read_data(log_filename):
    fp = open(log_filename)
    epochs = []
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for line in fp:
        if line.startswith("Epoch"):
            values = line.split(" ")
            epochs.append(float(values[1][:-1]))
            train_losses.append(float(values[4]))
            train_accuracies.append(float(values[7]))
            test_accuracies.append(float(values[10]))

    return epochs, train_losses, train_accuracies, test_accuracies


def create_plot(log_filename1, log_filename2, figure_file=""):
    epochs1, train_losses1, train_accuracies1, test_accuracies1 = read_data(log_filename1)
    epochs2, train_losses2, train_accuracies2, test_accuracies2 = read_data(log_filename2)
    if len(epochs1) > len(epochs2):
        epochs = epochs2
    else:
        epochs = epochs1

    train_accuracies1 = train_accuracies1[:len(epochs)]
    train_accuracies2 = train_accuracies2[:len(epochs)]
    train_losses1 = train_losses1[:len(epochs)]
    train_losses2 = train_losses2[:len(epochs)]
    test_accuracies1 = test_accuracies1[:len(epochs)]
    test_accuracies2 = test_accuracies2[:len(epochs)]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 6

    # numsims = []
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(3.4, 2))
    # lines = []
    # scn_names = []
    # for scn_name in all_data:
    #     mean_values = []
    #     std_values = []
    #     sem_values = []
    #     numsims = []
    #     for numsim in sorted(all_data[scn_name].keys()):
    #         mean_values.append(np.mean(all_data[scn_name][numsim]))
    #         std_values.append(np.std(all_data[scn_name][numsim]))
    #         sem_values.append(stats.sem(all_data[scn_name][numsim]))
    #         numsims.append(numsim / 1000.0)
    #     scn_names.append(scn_name)
    #     lines.append(ax.errorbar(numsims, mean_values, sem_values, linestyle='-', marker='o', capsize=4))
    #
    # min_x = min(numsims) - 2
    # max_x = max(numsims) + 2

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.set_xticks([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
    #ax.set_xticks([0, 3, 7, 10, 15, 20, 25, 50, 100])
    #ax.set_xlim(min_x, max_x)
    ax.set_xlabel(r'The Epoch')
    ax.set_ylabel('The Accuracy')
    #    ax.legend(lines, ['scenarios 1', 'scenarios 2', 'scenarios 3', 'scenarios 4', 'scenarios 5','scenarios 6'])
    #    ax.legend(lines, scn_names)
    #rect = patches.Rectangle((10.0, 0.0), 0.01, 10.0, linewidth=2, edgecolor='r', fill=False)
    #    rect = patches.Rectangle((0.9, 5.0), 0.2, 1.0, linewidth=2, edgecolor='r', fill=False)
    #ax.add_patch(rect)
    ax.plot(epochs, train_accuracies1)
    ax.plot(epochs, test_accuracies1)
    ax.plot(epochs, train_accuracies2)
    ax.plot(epochs, test_accuracies2)
    ax.legend(["Train Accuracy with Random", "Test Accuracy with Random", "Train Accuracy without Random", "Test Accuracy without Random"])

    if figure_file:
        plt.savefig(figure_file, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    create_plot("../topk_gat_test_notnull_training.log", "../topk_gat_test_notnull_notrandom_training.log", "test_training.pdf")
