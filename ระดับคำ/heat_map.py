import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

def make_heatmap(name_file,label1,label2,data):
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(label1)))
    ax.set_yticks(np.arange(len(label2)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(label1,fontname="Tahoma",fontsize=5)
    ax.set_yticklabels(label2,fontname="Tahoma",fontsize=5)

    ax.set_xlabel("sentence")
    ax.set_ylabel("Question")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")
    """
    # Loop over data dimensions and create text annotations.
    for i in range(len(label1)):
        for j in range(len(label2)):
            text = ax.text(j, i, data[i,j],
                        ha="center", va="center", color="w")
    """

    fig.tight_layout()
    plt.savefig(name_file)
    plt.clf()

