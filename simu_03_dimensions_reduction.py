from scipy.special import comb
'''
In a propagation, X -> Y, where X is subset of or equal to Y.
The number of possible Y is euqal to the number of subset of the supplementary set of X.
The connections number is usually (N**2 - N) /2 excluding mutual and self-to-self connections.
If the connections number << the number of possible Y, then it is reducing dimensions.
'''
print('%10s: %20s %20s %20s %20s %20s' % ('', 'propagations', 'connections', 'propa/conn', 'propa_comp', 'comp/propas'))
for N in range(2, 50): 
    y_number = 0
    for i in range(N + 1):
        y_number += comb(N, i)
    connections_number = N * (N - 1) / 2
    target_complexcity = 2 ** N
    # connection_complexity = N ** 2
    print('%10s: %20s %20s %20s %20s %20s' % (
        N, y_number, connections_number, round(y_number / float(connections_number), 0), target_complexcity, target_complexcity / float(y_number)))



