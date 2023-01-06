from typing import List, Union
from matplotlib import pyplot as plt
from numpy import ndarray


def heatmap(matrix: Union[List[List], ndarray], title="", xlabel="", file_path=None, fill=True, text_color=None):
    plt.clf()
    fig, ax = plt.subplots()
    ax.imshow(matrix)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            tc = text_color or ('black' if matrix[i][j] else 'white')
            ax.text(j, i, int(matrix[i][j]) if fill else '', ha="center", va="center", color=tc)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    if file_path is not None:
        plt.savefig(file_path)
    # plt.show()


def linear(id_weights: dict, title="", xlabel="", file_path=None):
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    for id, weight in id_weights.items():
        plt.plot(weight)
    if file_path is not None:
        plt.savefig(file_path)
    # plt.show()
