def sequential_weighted_avg(x, weights):
    """Return a sequence by weighted averaging of x (a sequence of vectors).
    Args:
        x: batch * len2 * hdim
        weights: batch * len1 * len2, sum(dim = 2) = 1
    Output:
        x_avg: batch * len1 * hdim
    """
    return weights.bmm(x)
