import pandas as pd
import numpy as np
from tools import make_distr

distrib = dict()
containers = pd.read_csv('DS_1.csv', header=0, sep=';')
containers['val'] = (containers['Container type'])**2 - 7/2*(containers['Container type']) + 7/2
distrib = make_distr(containers, 40000, 20, 1, 1, 1, 1, 1, 5000) #containers, Wp, L, bias, lambd_val, lambd_weight, number_of_exp, num_res_with_min_energy, num_reads