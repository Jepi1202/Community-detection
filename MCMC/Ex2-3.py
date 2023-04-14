import numpy as np
import time

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
        while links_as_arrays[0][j] < i:
            j += 1
        index[i] = j

    nodes_in_com1 = np.count_nonzero(x0 == 0) # /!\ Depends if x0 is given as 0/1 or 1/2
    return links_to_community, links_as_arrays[1], index, nodes_in_com1

def metroHast_K2(N, K, p, A1, A2, B, G, x0, Tmax, rng,
                 links_to_community, links_as_array, index, nodes_in_com1):
    links_to_com1, links_to_com2 = links_to_community
    log_probs = np.log((A1, A2, B, (1 - A1), (1 - A2), (1 - B)))
    random_u = np.log(rng.random(Tmax))
    random_slots = rng.choice(N, size=Tmax)
    x = x0
    for t, log_u, replaced_slot in zip(range(1, Tmax), random_u, random_slots):
        old_com_1 = (x[replaced_slot] == 0) # /!\
        n1, n2 = links_to_com1[replaced_slot], links_to_com2[replaced_slot]
        n1c, n2c = nodes_in_com1 - n1, N - nodes_in_com1 - n2
        log_alpha = np.sum(np.array([-n1, n2, n1 - n2, -n1c, n2c, n1c - n2c])*log_probs)
        if not old_com_1:
            log_alpha = -log_alpha

        if log_u < log_alpha:
            x[replaced_slot] = 1 if old_com_1 else 0 # /!\
            lower = index[replaced_slot]
            upper = N if replaced_slot + 1 >= N else index[replaced_slot + 1]
            nodes_to_update = links_as_array[lower:upper]
            increment = -1 if old_com_1 else 1
            links_to_com1[nodes_to_update] += increment
            links_to_com2[nodes_to_update] -= increment
            nodes_in_com1 += increment
    return x, links_to_community, nodes_in_com1

if __name__ == "__main__":
    t_start = time.time()
    # Filenames
    In = 'G.npy'
    Out = 'X.npy'

    # Parameters
    N = 16572
    K = 2
    p = [0.5, 0.5]
    a = 39.76
    b = 3.29
    A1 = A2 = a/N
    B = b/N

    T = 1000000 # number of iterations to perform
    rng = np.random.default_rng()

    G = np.load(In)
    print('Read done:', time.time() - t_start)
    x0 = rng.choice(K, size=N, p=p).astype(np.uint8) # /!\ If more than 256 communities
    links_to_community, links_as_array, index, nodes_in_com1 = pretreatement(N, G, x0)
    print('Pretreatment done:', time.time() - t_start)

    estimation, links_to_community, nodes_in_com1 = metroHast_K2(
            N, K, p, A1, A2, B, G, x0, T, rng, links_to_community, links_as_array, index, nodes_in_com1)

    np.save(Out, estimation + 1, allow_pickle=False)
    print('Done:', time.time() - t_start)
