import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import argrelextrema
from scipy.interpolate import griddata
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)


def phase_shift(data,idx_shift):
    data_new = np.zeros_like(data[:-1])
    data_new[idx_shift:] = data[:-idx_shift-1]
    data_new[0:idx_shift] = data[-idx_shift-1:-1]
    data_new_new = np.zeros_like(data)
    data_new_new[:-1] = data_new
    data_new_new[-1] = data_new[0]
    for i in range(np.shape(data_new_new)[1]):
        data_new_new[:,i] = gaussian_filter1d(data_new_new[:,i],2)
    data_new_new[-1] = data_new_new[0]
    return data_new_new

path_ = current_dir+"/opensource_dataset/"


idx_slope_end = 4
ra_slope = scio.loadmat(path_+"rampascent_incline.mat")['rampascent_incline'][0:idx_slope_end]
rd_slope = -scio.loadmat(path_+"rampdescent_incline.mat")['rampdescent_incline'][0:idx_slope_end]

ra_hq = scio.loadmat(path_+"rampascent_hip_theta.mat")["rampascent_hip_theta"]
ra_ht = scio.loadmat(path_+"rampascent_hip_tau.mat")["rampascent_hip_tau"]
ra_hq = phase_shift(ra_hq, 3)[:,0:idx_slope_end]
ra_ht = phase_shift(ra_ht, 3)[:,0:idx_slope_end]
ra_kq = -scio.loadmat(path_+"rampascent_knee_theta.mat")["ramp_ascent_theta"]
ra_kt = -scio.loadmat(path_+"rampascent_knee_tau.mat")["ramp_ascent_tau"]
ra_kq = phase_shift(ra_kq, 3)[:,0:idx_slope_end]
ra_kt = phase_shift(ra_kt, 3)[:,0:idx_slope_end]
ra_aq = scio.loadmat(path_+"rampascent_ankle_theta.mat")["rampascent_ankle_theta"]
ra_at = scio.loadmat(path_+"rampascent_ankle_tau.mat")["rampascent_ankle_tau"]
ra_aq = phase_shift(ra_aq, 3)[:,0:idx_slope_end]
ra_at = phase_shift(ra_at, 3)[:,0:idx_slope_end]

rd_hq = scio.loadmat(path_+"rampdescent_hip_theta.mat")["rampdescent_hip_theta"]
rd_ht = scio.loadmat(path_+"rampdescent_hip_tau.mat")["rampdescent_hip_tau"]
rd_hq = phase_shift(rd_hq, 6)[:,0:idx_slope_end]
rd_ht = phase_shift(rd_ht, 6)[:,0:idx_slope_end]
rd_kq = -scio.loadmat(path_+"rampdescent_knee_theta.mat")["yavg"]
rd_kt = -scio.loadmat(path_+"rampdescent_knee_tau.mat")["yavg"]
rd_kq = phase_shift(rd_kq, 6)[:,0:idx_slope_end]
rd_kt = phase_shift(rd_kt, 6)[:,0:idx_slope_end]
rd_aq = scio.loadmat(path_+"rampdescent_ankle_theta.mat")["rampdescent_ankle_theta"]
rd_at = scio.loadmat(path_+"rampdescent_ankle_tau.mat")["rampdescent_ankle_tau"]
rd_aq = phase_shift(rd_aq, 6)[:,0:idx_slope_end]
rd_at = phase_shift(rd_at, 6)[:,0:idx_slope_end]

s = scio.loadmat(path_+"speed.mat")["speed"][6:15:4]
s_hq = scio.loadmat(path_+"treadmill_hip_theta.mat")["treadmill_hip_theta"]
s_ht = scio.loadmat(path_+"treadmill_hip_tau.mat")["treadmill_hip_tau"]
s_hq = phase_shift(s_hq, 4)[:,:20][:,6:15:4]
s_ht = phase_shift(s_ht, 4)[:,:20][:,6:15:4]
s_kq = -scio.loadmat(path_+"treadmill_knee_theta.mat")["theta"]
s_kt = -scio.loadmat(path_+"treadmill_knee_tau.mat")["tau"]
s_kq = phase_shift(s_kq, 4)[:,:20][:,6:15:4]
s_kt = phase_shift(s_kt, 4)[:,:20][:,6:15:4]
s_aq = scio.loadmat(path_+"treadmill_ankle_theta.mat")["treadmill_ankle_theta"]
s_at = scio.loadmat(path_+"treadmill_ankle_tau.mat")["treadmill_ankle_tau"]
s_aq = phase_shift(s_aq, 4)[:,:20][:,6:15:4]
s_at = phase_shift(s_at, 4)[:,:20][:,6:15:4]

idx = np.arange(0,101)

v_key = ["0.8","1.0","1.2"]
s_key = ["-10","-8","-5", "0", "5","8","10"]
def load_data(vk, sk, m=60):
    exists = (vk in v_key) and (sk in s_key)
    if not exists:
        print("Velocity or Slope not exists")
        print(v_key)
        print(s_key)
        return None
    if sk == "0": # variable speed
        i = v_key.index(vk)
        kq = s_kq[:,i]-10
        kt = -s_kt[:,i]*m
        aq = s_aq[:,i]
        at = -s_at[:,i]*m
        return np.vstack((kq, kt, aq, at))
    elif sk[0] == "-": # downslope
        if vk != "1.0":
            print("Velocity not exists")
            return None
        i = s_key.index(sk)
        kq = rd_kq[:,i]-10
        kt = -rd_kt[:,i]*m
        aq = rd_aq[:,i]
        at = -rd_at[:,i]*m
        return np.vstack((kq, kt, aq, at))
    else:
        if vk != "1.0":
            print("Velocity not exists")
            return None
        i = s_key.index(sk)-4
        kq = ra_kq[:,i]-10
        kt = -ra_kt[:,i]*m
        aq = ra_aq[:,i]
        at = -ra_at[:,i]*m
        return np.vstack((kq, kt, aq, at))





