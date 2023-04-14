import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import eig

# Parameters for figure formatting
#matplotlib.use("pgf")
sns.set_theme(context='paper', style='darkgrid', palette='deep')
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.serif' : ['Computer Modern Roman'],
    'text.usetex': True,
    'pgf.rcfonts': True,
    'axes.labelsize': 12,
    'axes.labelpad': 15,
    'font.size': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.formatter.use_mathtext': True,
    #'axes.formatter.limits': (-0, 0)
})

# Maximum time step
TMAX = 30

# Transition matrix
Q = np.array([[0, 0.1, 0.1, 0.8],
	[1, 0, 0, 0],
	[0.6, 0, 0.1, 0.3],
	[0.4, 0.1, 0.5, 0]])

Qt = Q.T

# Initial distributions
# X0, filename = [0.25, 0.25, 0.25, 0.25], '1-1-uniform.pgf'
X0, filename = [0, 0, 1, 0], '1-1-state3.pgf'

# Array initialisation
chain = np.empty([TMAX, len(X0)])
chain[0] = X0

# Array filling
for t in range(TMAX - 1):
    chain[t+1] = Qt @ chain[t, :]

# Data formatting for figure
df = pd.DataFrame(data=chain, columns=['$x_1$', '$x_2$', '$x_3$', '$x_4$'])
tidy = df.stack().reset_index().rename(columns={"level_0": "Pas de temps $t$", "level_1": "États", 0: "Probabilités $P(X_t = x_i)$"})
sns.lineplot(x='Pas de temps $t$', y='Probabilités $P(X_t = x_i)$', hue='États', data=tidy)
plt.show()
#plt.savefig(filename, bbox_inches='tight')

# Matrix power
print(np.linalg.matrix_power(Q, 1000))

# Stationnary distribution
print(chain[TMAX - 1])
