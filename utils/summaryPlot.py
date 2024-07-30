def summary_plot(results, ax, col='loss', valid_legend='Validation', training_legend='Training', ylabel='Loss', fontsize=20):

    for(column, color, label) in zip([f'train_{col}_epoch', f'valid_{col}'], ['black', 'red'], [training_legend, valid_legend]):
        results.plot(x='epoch', y=column, label=label, marker='o',color=color, ax=ax)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    return ax