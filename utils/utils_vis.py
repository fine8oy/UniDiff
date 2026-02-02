import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from scipy.spatial import distance


def prepare_data(ori_sig, gen_sig):
    # Analysis sample size (for faster computation)
    sample_num = min([1000, len(ori_sig)])
    idx = np.random.permutation(len(ori_sig))[:sample_num]

    # Data preprocessing
    # ori_ssig = np.asarray(ori_sig)
    # generated_data = np.asarray(gen_sig)

    ori_sig = ori_sig[idx]
    gen_sig = gen_sig[idx]
    no, seq_len, dim = ori_sig.shape
    prep_ori = np.reshape(np.mean(ori_sig[0, :, :], 1), [1, seq_len])
    prep_gen = np.reshape(np.mean(gen_sig[0, :, :], 1), [1, seq_len])
    for i in range(1, sample_num):
        prep_ori = np.concatenate((prep_ori,
                                    np.reshape(np.mean(ori_sig[i, :, :], 1), [1, seq_len])))
        prep_gen = np.concatenate((prep_gen,
                                        np.reshape(np.mean(gen_sig[i, :, :], 1), [1, seq_len])))
    return prep_ori, prep_gen, sample_num





