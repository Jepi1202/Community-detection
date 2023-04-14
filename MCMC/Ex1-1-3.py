import numpy as np
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
    'axes.labelpad': 15,
    'font.size': 15,
    'legend.fontsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.formatter.use_mathtext': True,
    #'axes.formatter.limits': (-0, 0)
})

# Parameters
T = 2000
Q = np.array([[0, 0.1, 0.1, 0.8],
	[1, 0, 0, 0],
	[0.6, 0, 0.1, 0.3],
	[0.4, 0.1, 0.5, 0]])

# Array initialisations
pi = np.array([1, 0, 0, 0]) # Current distribution
chain = np.empty(T) # Stores the states that are chosen as realisation of the chain
occurences = np.empty([T + 1, 4]) # Stores the number of occurences of each state at each time step
occurences[0] = [0, 0, 0, 0]
rng = np.random.default_rng()

for i in range(T):
    sample = rng.multinomial(1, pi) # Pick a random state based on current distribution
    occurences[i] += sample
    occurences[i + 1] = occurences[i]
    chain[i] = np.nonzero(sample)[0][0]
    pi = sample@Q

for i in range(T):
    occurences[i] /= i + 1 # Convert data from number of occurences to occurence frequencies

# Data formatting for occurence frequencies plot
df = pd.DataFrame(data=occurences[:T], columns=['$x_1$', '$x_2$', '$x_3$', '$x_4$'])
tidy = df.stack().reset_index().rename(columns={"level_0": "Pas de temps $t$", "level_1": "États", 0: "Taux d'apparition"})
sns.lineplot(x='Pas de temps $t$', y="Taux d'apparition", hue='États', data=tidy)
plt.show()
plt.savefig('1-1-3-freq.pgf', bbox_inches='tight')
plt.cla()

# Chain plot
plt.plot(chain[:50])
plt.xlabel('Pas de temps $t$')
plt.ylabel('État')
plt.yticks(ticks=range(4), labels=['$x_1$', '$x_2$', '$x_3$', '$x_4$'])
plt.show()
plt.savefig('1-1-3-chain.pgf', bbox_inches='tight')



