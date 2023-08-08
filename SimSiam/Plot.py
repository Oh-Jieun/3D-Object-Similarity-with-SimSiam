from matplotlib import patches
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np

def scatter(x, labels):
    num_classes = 20
    palette = np.array(sns.color_palette("hls", num_classes)) # Choosing color
    fig, ax = plt.subplots()

    ## Create a seaborn scatter plot ##
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)], marker='o')
    
    idx2name = ['plane', 'bethtub', 'bed', 'bench', 'boat', 'bus', 'cabinet', 'car', 'chair', 'clock', 'display', 'faucet', 'guitar', 'lamp', 'speaker', 'rifle', 'sofa', 'table', 'phone', 'vessel']

    # Create legend handles
    handles = [patches.Patch(color=palette[i], label=idx2name[i]) for i in range(num_classes)]
    
    # Add legend to the plot
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0., title='Classes')

    ax.grid(True)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)    
    plt.xlabel('xlabel')
    plt.ylabel('ylabel')