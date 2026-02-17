import matplotlib.pyplot as plt
import seaborn as sns

def setup_matplotlib_styles():
    plt.rc('text', usetex=True)
    plt.rc('font', family='Nimbus Roman')

    SMALL_SIZE = 9
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 9

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    plt.rc('ytick', direction='in')
    plt.rc('xtick', direction='in')

    