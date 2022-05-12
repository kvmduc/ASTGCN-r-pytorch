import numpy as np
import glob


temp = np.load('./data/PEMS04/PEMS04_r1_d0_w0_astcgn.npz' , allow_pickle=True)
list = temp.files
for element in list:
    print(element, temp[element].shape)