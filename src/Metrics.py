import numpy as np
from scipy.stats import entropy, mode

"""
Contains metrics for shot type determination
"""


def bi_entropy(histogram):
    # Calcul des probabilités marginales par la formule des probabilités totales
    x_probs = np.sum(histogram, axis=1)
    y_probs = np.sum(histogram, axis=0)
    x_entropy = entropy(x_probs)  # entropy de scipy.stats gère la normalisation
    y_entropy = entropy(y_probs)
    return x_entropy, y_entropy


def bi_amplitude(flow):
    x_amplitude = flow[:, :, 0].max() - flow[:, :, 0].min()
    y_amplitude = flow[:, :, 1].max() - flow[:, :, 1].min()
    return x_amplitude, y_amplitude


def bi_mean(flow):
    x_mean = np.mean(flow[:, :, 0])
    y_mean = np.mean(flow[:, :, 1])
    return x_mean, y_mean


def bi_max(flow):
    x_max = flow[:, :, 0].max()
    y_max = flow[:, :, 1].max()
    return x_max, y_max


def bi_min(flow):
    x_min = flow[:, :, 0].min()
    y_min = flow[:, :, 1].min()
    return x_min, y_min


def bi_mode(histogram):  # Non-fonctionnel
    x_mode = mode(histogram, axis=1).mode[0]
    y_mode = mode(histogram, axis=0).mode[0]
    return x_mode, y_mode


# Calcul à partir de l'histogramme et du flow par facilité des formules
# Mais toutes ces métriques sont calculables via l'histogramme seul si réellement nécessaire
def get_X_vector(flow, histogram):
    Vx_entropy, Vy_entropy = bi_entropy(histogram)
    Vx_amplitude, Vy_amplitude = bi_amplitude(flow)
    Vx_max, Vy_max = bi_max(flow)
    Vx_min, Vy_min = bi_min(flow)
    Vx_mean, Vy_mean = bi_mean(flow)
    # Vx_mode, Vy_mode = bi_mode(histr)
    return [Vx_entropy, Vy_entropy, Vx_amplitude, Vy_amplitude, Vx_max, Vy_max, Vx_min, Vy_min, Vx_mean, Vy_mean]
