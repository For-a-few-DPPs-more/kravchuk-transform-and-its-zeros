import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.random as npr
import statsmodels.stats.proportion as stms
from chirp_signals import display_signal, the_chirp, the_noisy_chirp, the_white_noise
from detection_test import the_test, the_test_statistic
from kravchuk_display import planar_display, signal_display, spherical_display
from kravchuk_transform import the_transform, the_zeros
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat, savemat
from spherical_statistics import (
    empirical_F,
    empirical_K,
    the_distance,
    the_F_statistics,
    the_K_statistics,
)
from stft_transform import stft_display, the_stft_transform, the_stft_zeros
from white_noises import noise_samples

mpl.rcParams["xtick.labelsize"] = 30
mpl.rcParams["ytick.labelsize"] = 30
mpl.rcParams["axes.titlesize"] = 30
plt.rc("axes", labelsize=35)
plt.rc("legend", fontsize=30)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
mpl.rcParams["font.family"] = "roman"

sapin = (0.0353, 0.3216, 0.1569)
carmin = (0.7294, 0.0392, 0.0392)
bleu = (0.2, 0.2, 0.7020)
rose = (0.56, 0.004, 0.32)
marron = (0.51, 0.3, 0.11)
jaune = (0.93, 0.69, 0.13)
