import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Parameters for figure formatting
matplotlib.use("pgf")
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

# Graph generator for K=2
def SBM_K2(N, K, p, A, B, rng):
    x = rng.choice(2, N, p=p).astype(np.uint8) # /!\ If more than 256 communities
    G = np.zeros([N, N], dtype=np.uint8)
    for i in range(N - 1):
        for j in range(i + 1, N):
            rand = rng.random()
            if x[i] == x[j]:
                if rand <= A:
                    G[i, j] = G[j, i] = 1
            else:
                if rand <= B:
                    G[i, j] = G[j, i] = 1
    return (x, G)

# Computes concordance between true and estimation community vectors for K=2
def concordance_K2(N, K, true, estimation):
    N1 = np.count_nonzero(true == estimation)
    N2 = np.count_nonzero(true == np.mod(estimation + 1, 2))
    return max(N1, N2)/N

"""
Returns the data structures based on G necessary to the Metropolis-Hastings implementation
    -links_to_community is an 2*N table indicating how many nodes in community 1 and 2 are linked
    to each individual node, according to the current community repartition

    -links_as_arrays and index allow to access the list of all nodes linked to an individual node i
    trough links_as_arrays[1][index[i]:index[i+1]]

    -nodes_in_com1 is simply the number of nodes in community 1 according to the current node
    repartition

Refer to the report for more information
"""
def pretreatement(N, G, x0):
    links_to_community = np.zeros([2, N])
    links_as_arrays = np.nonzero(G)
    temp = np.vstack((x0[links_as_arrays[1]], links_as_arrays[0]))
    temp = np.unique(temp.T, axis=0, return_counts=True)
    links_to_community[tuple(temp[0].T)] = temp[1]

    index = np.empty(N, dtype=int)
    j = 0
    for i in range(N):
        while links_as_arrays[0][j] < i and j < len(links_as_arrays[0]) - 1:
            j += 1
        index[i] = j

    nodes_in_com1 = np.count_nonzero(x0 == 0) # /!\ Depends if x0 is given as 0/1 or 1/2
    return links_to_community, links_as_arrays[1], index, nodes_in_com1

def metroHast_K2(N, K, p, A, B, G, x0, Tmax, rng,
                 links_to_community, links_as_array, index, nodes_in_com1):

    chain = np.empty([Tmax, N], dtype=np.uint8)
    links_to_com1, links_to_com2 = links_to_community
    log_r, log_rc = np.log((A/B, (1 - A)/(1 - B)))
    random_u = np.log(rng.random(Tmax))
    random_slots = rng.choice(N, size=Tmax)
    random_communities = rng.choice(K, size=Tmax, p=p)
    chain[0] = x0
    for t, log_u, replaced_slot, new_com in zip(range(1, Tmax), random_u, random_slots, random_communities):
        chain[t] = chain[t - 1]
        old_com = chain[t][replaced_slot]
        if old_com != new_com:
            old_com_1 = (old_com == 0) # /!\
            n = links_to_com1[replaced_slot] - links_to_com2[replaced_slot]
            nc= 2*nodes_in_com1 - N - n
            log_alpha = n*log_r + nc*log_rc
            if old_com_1:
                log_alpha = -log_alpha

            if log_u < log_alpha:
                chain[t][replaced_slot] = 1 if old_com_1 else 0 # /!\
                lower = index[replaced_slot]
                upper = N if replaced_slot + 1 >= N else index[replaced_slot + 1]
                nodes_to_update = links_as_array[lower:upper]
                increment = -1 if old_com_1 else 1
                links_to_com1[nodes_to_update] += increment
                links_to_com2[nodes_to_update] -= increment
                nodes_in_com1 += increment
    return chain

# Plots the evolution of concordance along the Metropolis-Hastings chain
def show_convergence(N, K, p, A, B, Tmax, rng):
    x, G = SBM_K2(N, K, p, A, B, rng)
    x0 = rng.choice(K, size=N, p=p)
    links_to_community, links_as_array, index, nodes_in_com1 = pretreatement(N, G, x0)
    chain = metroHast_K2(N, K, p, A, B, G, x0, Tmax, rng, links_to_community, links_as_array, index, nodes_in_com1)
    data = np.empty(Tmax)
    for i in range(Tmax):
        data[i] = concordance_K2(N, K, x, chain[i, :])

    plt.plot(data)
    plt.xlabel("Nombre d'itérations")
    plt.ylabel('Concordance $A(x*, \hat{x})$')
    plt.show()
    plt.savefig('convergence.pgf', bbox_inches='tight')


if __name__ == "__main__":
    N = 5000
    K = 2
    p = [0.5, 0.5]
    Tmax = 100000
    queue = int(0.75*Tmax)
    rng = np.random.default_rng()

    """
    mean_degree = 10
    r = 0.2
    a = (2*mean_degree)/(1 + r)
    b = 2*mean_degree - a
    A = a/N
    B = b/N
    A = 0.04
    B = 0.005
    show_convergence(N, K, p, A, B, Tmax, rng)
    """
    Ngraphs = 30
    mean_degree = 10
    data_x = np.linspace(0.00001, 1, num=200)
    data_y = np.empty(len(data_x))
    condition = np.empty(len(data_x))
    limit = 0
    for k in range(len(data_x)):
        r = data_x[k]
        a = (2*mean_degree)/(1 + r)
        b = 2*mean_degree - a
        A = a/N
        B = b/N
        condition[k] = (a - b)**2 > 2*(a + b)
        if (a - b)**2 <= 2*(a + b) and limit == 0:
            limit = r
        Egraphs = 0
        for i in range(Ngraphs):
            x, G = SBM_K2(N, K, p, A, B, rng)
            x0 = rng.choice(K, N, p=p)
            links_to_community, links_as_array, index, nodes_in_com1 = pretreatement(N, G, x0)
            chain = metroHast_K2(N, K, p, A, B, G, x0, Tmax, rng, links_to_community, links_as_array, index, nodes_in_com1)
            Edistribtion = 0
            for estimation in chain[queue:]:
                Edistribtion += concordance_K2(N, K, x, estimation)
            Edistribtion /= len(chain[queue:])
            Egraphs += Edistribtion
        data_y[k] = Egraphs/Ngraphs
        print(k)

    print(data_x)
    print(condition)
    plt.plot(data_x, data_y)
    plt.vlines(limit, min(data_y), max(data_y), linestyles='dashed', colors='orange')
    plt.xlabel('Rapport $b/a$')
    plt.ylabel('Concordance moyenne')
    plt.show()
    plt.savefig('2-2-test.pgf', bbox_inches='tight')
