# This part if for making a movie from the Brutus output

import numpy as np
import matplotlib
import h5py
import os
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_data_to_hdf5 import *

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

lw_tol_vec = iterate_sim_data_dir()
print('return lw_tol_vec', lw_tol_vec)
# txt_to_h5py('./output/per_tstep_e-1_lw_136_tol_24_output_test.txt', 'per_tstep_e-1_lw_136_tol_24_output_test.h5part')
fig = plt.figure(dpi=200)

# h5f = h5py.File('../output/sim_data_test.hdf5', 'r')
# h5g = h5f['/home/pqian/simon_brutus/sim_data_test']  # root for hdf5 file

x_time, y_seperation, label = generate_x_y(lw_tol_vec)

plot_handles = []
for i in range(len(x_time)):
    # Plot
    print('label', label[i])
    p1 = plt.semilogy(x_time[i], y_seperation[i],
                      label=r'$\epsilon$=$10^{-%d}$,$L_w=%d$'
                            % (label[i][0], label[i][1]))  # 'k-o')
#     p2=plt.semilogy(x_time2, y_seperation2,
#  label=r'$\epsilon$=$10^{-8}$,$L_w=72$')#, 'k-o')#, markersize=30)
    plot_handles.append(p1[0])

plt.legend(handles=plot_handles)

# plt.plot(x_time, y_seperation, 'ko', markersize=30)
plt.xlabel('Time')
plt.ylabel(r'Seperation')
plt.xlim(0, 100)
# plt.ylim(-12, 2)
plt.savefig('separation.pdf')
print('figure saved!')
