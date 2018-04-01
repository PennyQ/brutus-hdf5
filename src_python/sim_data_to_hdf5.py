# This part is for transform the text file format to hdf5
# so we can display in visualization softwares such as Paraview

import numpy as np
import h5py
import os


def read_txt_to_hdf(sim_name, txt_file, h5f):
    # file-input.py
    f = open(txt_file, 'r')
    message = f.readlines()
    t_list = []
    vector = []
    evolve_t = []

    for line in message:
        if 'tolerance' in line:
            continue
        if 't' in line:
            if '=' in line:
                t_list.append(float(line.split('=')[1]))
            else:
                evolve_t = line.split(':')[1]
            continue
        # for lines with vector information
        vector.append(line.split(',')[:-1])
    print('vector', len(vector))
    print('vector', vector[0])

    try:
        vector = np.array(vector).astype(np.float)
    except ValueError:
        # last element can be an empty list
        vector = np.array(vector[:-1]).astype(np.float)

    n_particles = 3
    try:
        h5g = h5f.create_group(sim_name)
        # i~(0-10000), t~(0.01-100.01)
        for i in range(len(t_list)):
            step_name = 'Step#%d' % i
            step_h5g = h5g.create_group(step_name)
            step_h5g.create_dataset('Mass', data=vector[i*3:i*3+3, 0])
            step_h5g.create_dataset('X', data=vector[i*3:i*3+3, 1])
            step_h5g.create_dataset('Y', data=vector[i*3:i*3+3, 2])
            step_h5g.create_dataset('Z', data=vector[i*3:i*3+3, 3])
            step_h5g.create_dataset('VX', data=vector[i*3:i*3+3, 4])
            step_h5g.create_dataset('VY', data=vector[i*3:i*3+3, 5])
            step_h5g.create_dataset('VZ', data=vector[i*3:i*3+3, 6])
        print('write to hdf5 done!')
    except ValueError:
        print('Group %s already exists, skip and go to next group!' % sim_name)


# TODO: define input_dir and output_dir set by user
def iterate_sim_data_dir(input_dir='../output/sim_data_test',
                         output_hdf_path='../output/sim_data_test.hdf5'):
    # output hdf file
    try:
        h5f = h5py.File(output_hdf_path, 'a')
    except IOError:
        h5f = h5py.File(output_hdf_path, 'r+')
    # length width and tolerance
    lw_tol_vec = []

    # Save all output from the sim_data into one HDF5 file
    for each_sim in os.listdir(input_dir):
        each_sim_path = os.path.join(input_dir, each_sim)
        print('each_sim name', each_sim)
        txt_file_path = os.path.join(each_sim_path, 'output.txt')

        if os.path.isfile(txt_file_path):
            # extract lw and tolerance from simulation dir name
            lw_tol_vec.append([int(each_sim.split('=')[1].split('_')[0]),
                               int(each_sim.split('=')[2].split('_')[0])])
            read_txt_to_hdf(each_sim, txt_file_path, h5f)
    lw_tol_vec = np.array(lw_tol_vec).astype(int)
    print('lw_tot_vec', lw_tol_vec)

    h5f.close()
    return lw_tol_vec


# TODO: define h5f_path define by user
def generate_x_y(lw_tol_vec, h5f_path='../output/sim_data_test.hdf5'):
    # This part is for generating Figure 1 on [P. Zwart & Boekholt, 2018]
    x = []
    y = []
    label = []
    h5f = h5py.File(h5f_path, 'r')
    for lw_tol in lw_tol_vec:
        # TODO: read from new hdf file structure
        for each_key in h5f['unp_lw=%d_tol=%d_N=3' % (lw_tol[0], lw_tol[1])].keys():
            print('each_key', each_key)
            # Unper
            unp_t = h5f['unp_lw=%d_tol=%d_N=3/Step#' % (lw_tol[0], lw_tol[1])].value
            # evolve_t1 = h5f['unp_lw=%d_tol=%d_N=3/evolve_t' % (lw_tol[0], lw_tol[1])].value
            unp_x = h5f['unp_lw=%d_tol=%d_N=3/X' % (lw_tol[0], lw_tol[1])].value
            unp_y = h5f['unp_lw=%d_tol=%d_N=3/Y' % (lw_tol[0], lw_tol[1])].value
            unp_z = h5f['unp_lw=%d_tol=%d_N=3/Z' % (lw_tol[0], lw_tol[1])].value
            unp_xyz = np.array([unp_x, unp_y, unp_z]).T
            # Per
            per_t = h5f['per_lw=%d_tol=%d_N=3/Step#' % (lw_tol[0], lw_tol[1])].value
            # evolve_t2 = h5f['per_lw=%d_tol=%d_N=3/evolve_t' % (lw_tol[0], lw_tol[1])].value
            per_x = h5f['per_lw=%d_tol=%d_N=3/X' % (lw_tol[0], lw_tol[1])].value
            per_y = h5f['per_lw=%d_tol=%d_N=3/Y' % (lw_tol[0], lw_tol[1])].value
            per_z = h5f['per_lw=%d_tol=%d_N=3/Z' % (lw_tol[0], lw_tol[1])].value
            per_xyz = np.array([per_x, per_y, per_z]).T
            #t1, evolve_t1, vector1 = read_to_numpy('./output/unp_lw=%d_tol=%d_N=3.txt'%(lw_tol[0], lw_tol[1]))
            #t2, evolve_t2, vector2 = read_to_numpy('./output/per_lw=%d_tol=%d_N=3.txt'%(lw_tol[0], lw_tol[1]))
            print('unp_t shape', len(unp_t))
            # Euclidean distance calculation for the two-solution seperation
            # TODO: 30000 here depends on total time step and dt
            re1 = np.linalg.norm(unp_xyz-per_xyz, axis=1)  # [0:30000]  # numpy.linalg.norm(array[x0,y0,z0] - array[x1,y1,z1])
            # print('hi',len(re1),re1[0:30000])
            re1 = re1.reshape((len(unp_t), 3))
            y.append(np.sum(re1, axis=1))
            x.append(unp_t[0:len(unp_t)])
            label.append(lw_tol)
    h5f.close()
    return x, y, label
