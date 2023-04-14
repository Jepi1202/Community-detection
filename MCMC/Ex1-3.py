import numpy as np
from scipy.stats import binom
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Parameters for figure formatting
matplotlib.use("pgf")
sns.set_theme(context='paper', style='darkgrid', palette='deep')
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.serif' : ['Computer Modern Roman'],
    'text.usetex': True,
    'pgf.rcfonts': True,
    'axes.labelsize': 24,
    'axes.labelpad': 10,
    'font.size': 15,
    'legend.fontsize': 15,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.formatter.use_mathtext': True,
    #'axes.formatter.limits': (-0, 0)
})

# Parameters
rng = np.random.default_rng()
K = 10
p = 0.3
r = 0.5
initstate = 7
Tmax = 1000

def q(y, x):
    if(x == 0 and y == 0):
        return r
    elif(x == K and y == K):
        return 1 - r
    elif(0 < x <= K and y == x - 1):
        return r
    elif(0 <= x < K and y == x + 1):
        return 1 - r
    else:
        return 0

if __name__ == "__main__":
    x = np.empty(Tmax)
    means = np.empty(Tmax)
    variances = np.empty(Tmax)
    x[0] = initstate
    means[0] = initstate
    variances[0] = 0
    y = 0
    for t in range(1, Tmax):
        prevx = x[t - 1]
        if(rng.random() <= r):
            if(prevx > 0):
                y = prevx - 1
            else:
                y = 0
        else:
            if(prevx < K):
                y = prevx + 1
            else:
                y = K

        alpha = (binom.pmf(y, K, p)*q(prevx, y))/(
                binom.pmf(prevx, K, p)*q(y, prevx))
        if(alpha > 1):
            alpha = 1

        if(rng.random() < alpha):
            x[t] = y
        else:
            x[t] = prevx

        means[t] = np.mean(x[:t + 1])
        variances[t] = np.var(x[:t + 1])

    frequencies = np.unique(np.append(x, range(K + 1)), return_counts=True)[1]
    frequencies = (frequencies - 1)/Tmax


    # Plot chain
    plt.plot(range(500), x[:500])
    plt.xlabel('Pas de temps $t$')
    plt.ylabel('$k$')
    plt.show()
    plt.savefig('1-3-chain-' + str(r)[2] + '.pgf', bbox_inches='tight')
    plt.cla()

    # Plot mean
    plt.plot(range(Tmax), means)
    plt.hlines(3, 0, Tmax, linestyles='dashed', colors='orange')
    plt.xlabel('Pas de temps $t$')
    plt.ylabel('$k$')
    plt.show()
    plt.savefig('1-3-mean-' + str(r)[2] + '.pgf', bbox_inches='tight')
    plt.cla()

    # Plot variance
    plt.plot(range(Tmax), variances)
    plt.hlines(2.1, 0, Tmax, linestyles='dashed', colors='orange')
    plt.xlabel('Pas de temps $t$')
    plt.ylabel('$k$')
    plt.show()
    plt.savefig('1-3-var-' + str(r)[2] + '.pgf', bbox_inches='tight')


    """ Messy figure
    df1 = pd.DataFrame(data=np.array([x, means, variances]).T,
                       columns=['chaîne', 'moyenne', 'variance'])
    tidy1 = df1.stack().reset_index().rename(
            columns={"level_0": "Pas de temps $t$", "level_1": "Statistiques", 0: "$k$"})
    sns.lineplot(x='Pas de temps $t$', y='$k$', hue='Statistiques', data=tidy1)
    plt.hlines(3, 0, Tmax, linestyles='dashed')
    plt.show()
    #plt.savefig('1-3-realisation-01.pgf')
    plt.cla()
    """
    # Plot histogram
    df2 = pd.DataFrame(
            data=np.array([frequencies, binom.pmf(range(K + 1), K, p)]).T,
            columns=['Fréquence observée', 'PMF théorique'])
    tidy2 = df2.stack().reset_index().rename(
        columns={"level_0": "$k$", "level_1": "", 0: "Fréquence"})
    ax = sns.barplot(x='$k$', y='Fréquence', hue='', data=tidy2)
    plt.show()
    plt.savefig('1-3-histogram-05-1000.pgf', bbox_inches='tight')
