import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def sample_a(chain, data):
    """Sample a given b and the observed chain.
    Chain contains the observed a_s and b_s
    """
    return st.gamma(n + 1, np.sum(data))
