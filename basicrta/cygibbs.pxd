#from basicrta.functions import norm_exp
from tqdm import tqdm
import numpy as np
cimport numpy as cnp
import gc
import cython
gc.enable

cnp.import_array()
#DTI = np.int64
#DTF = np.float64
#ctypedef cnp.int64_t DTI_t
#ctypedef cnp.float64_t DTF_t
cdef cygibbs_sorted
