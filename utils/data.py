import numpy as np


def ratios_to_integers(m, ratios):
    """
    subset_sizes = ratios_to_integers(m, ratios)

    Determines the size of each subset to be created from a dataset with a given size.

    :param m: Number of dataset samples
    :param ratios: list of ratios a dataset needs to split into.
                Each ratio should be in range (0,1) and all ratios
                must sum up to 1
    :return: list containing the sample size for each subset. Has same length as 'ratios'.
    """

    assert sum(ratios) == 1
    sizes = np.array(ratios)*m

    # Add the reminder to the first set
    while sum(sizes) < m:
        sizes[0] += 1

    return sizes.astype(np.int64).tolist()
