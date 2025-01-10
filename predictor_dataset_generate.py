import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import argrelextrema
from scipy.interpolate import griddata, interp1d
from greggdataset_loader import load_data_with_thigh
from greggdataset_loader import v_key, s_key
from utils import *
from plot_utils import IJRR_Imp_Cal
import warnings
from contextlib import redirect_stdout
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)


def function_that_warns():
    warnings.warn("This is a warning!")

m = 60

v_key = ["0.8","1.0","1.2"]
s_key = ["-5", "0", "5"]

# ab_list = ["AB01", "AB02", "AB03", "AB05", "AB06", "AB07", "AB08", "AB09", "AB10"]

ab_list = ["AB02", "AB03", "AB05", "AB06", "AB10"]



imp_dataset = []

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

fig2 = plt.figure()
ax = fig2.add_subplot(111)

path_ = current_dir+"/gregg_dataset/result/"

for ab_k in range(len(ab_list)):
    ab_info = np.load(path_+ab_list[ab_k]+"/ab_info.npy")
    print(ab_info)


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")  # 捕获所有警告
    function_that_warns()  # 触发警告

    # 检查是否有警告被触发
    if len(w) > 0:
        for warning in w:
            print(f"A warning was captured: {warning.message}")

def check_avaliable(k_mat, q_e_mat, slope=0):
    if k_mat[0,0] < 55 or k_mat[0,0]>180:
        return False
    if k_mat[0,1] < 45 or k_mat[0,1]>180:
        return False
    if (q_e_mat[1,1]<2 or q_e_mat[1,1]>20) and slope>=0:
        return False
    return True

def plot_st_division_point(q_k, q_a, t_k, t_a, ax1, ax2):
    # st1 knee
    p_1 = np.argmax(t_k[:25])
    p_0 = np.argmin(t_k[:25])
    if p_0 - p_1 == 0:
        p_0 = p_1 -10
    ax1.scatter([q_k[p_0], q_k[p_1]], [t_k[p_0], t_k[p_1]], color='r')
    # st1 ankle
    p_1 = np.argmin(t_a[0:15])
    p_2 = p_1 + 15
    p_1 = p_2 - 5
    ax2.scatter([q_a[p_1], q_a[p_2]], [t_a[p_1], t_a[p_2]], color='r')
    # st2 knee
    idx_max_t_k = np.argmax(t_k[10:30])+10
    local_min_list = argrelextrema(t_k[idx_max_t_k:60], np.less)[0]
    if len(local_min_list)>1:
        for i in local_min_list:
            if i > 10 and i < 30:
                p_1 = local_min_list[i]+idx_max_t_k
    else:
        p_1 =local_min_list[0] +idx_max_t_k
    p_2 = p_1 + 10
    ax1.scatter([q_k[p_1], q_k[p_2]], [t_k[p_1], t_k[p_2]], color='g')
    # st2 ankle
    p_1 = np.argmax(t_a)
    p_0 = p_1 - 10
    idx_max_t_a = np.argmax(t_a)
    p_2 = np.argmin(q_a[idx_max_t_a:])+idx_max_t_a
    ax2.scatter([q_a[p_1], q_a[p_2]], [t_a[p_1], t_a[p_2]], color='g')
    
def cal_gait_fea_thigh(q_t):
    fea = np.zeros((4,))
    P1, p_P1 = q_t[0], 0
    P2, p_P2 = np.min(q_t[30:80]), np.argmin(q_t[30:80])+30
    P3, p_P3 = q_t[-1], 99
    fea = np.array([P1, P2, P3,p_P1,p_P2,p_P3]).reshape((-1,))
    return fea


for ab_k in range(len(ab_list)):
    ab_info = np.load(path_+ab_list[ab_k]+"/ab_info.npy")
    # m, h = ab_info[-2], ab_info[-1]*0.001
    m, h = 75, ab_info[-1]*0.001
    for i in range(len(v_key)):
        for j in range(len(s_key)):
            data, mean_data = load_data_with_thigh(ab_k=ab_list[ab_k], vk=v_key[i], sk=s_key[j], m=m, show=False, ax=None)
            dt = grid_interp_dt(v=float(v_key[i]), s=float(s_key[j]), num_frame=mean_data.shape[1])
            print(v_key[i], s_key[j])
            for k in range(data.shape[2]):
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")  # 捕获所有警告
                        q_t, q_k, t_k, q_a, t_a = data[0,:,k], data[1,:,k], data[2,:,k], data[3,:,k], data[4,:,k]
                        # ax1.plot(q_k, t_k)
                        # ax2.plot(q_a, t_a)
                        # 计算Impedance
                        imp_cal = IJRR_Imp_Cal(q_k, t_k, q_a, t_a, dt, headless=True)
                        imp_cal.cal_imp_IJRR()
                        k_mat = imp_cal.k_mat
                        b_mat = imp_cal.b_mat
                        q_e_mat = imp_cal.q_e_mat
                        print(np.round(np.array(k_mat)*30,2))
                        print(np.round(np.array(b_mat)*30,2))
                        print(np.round(np.array(q_e_mat),2))
                        print("-----------------")
                        # plot_st_division_point(q_k, q_a, t_k, t_a, ax1, ax2)
                        # plt.show(block=False)
                        avaliable = check_avaliable(np.array(k_mat)*30, np.array(q_e_mat),slope=float(s_key[j]))
                        if avaliable:
                            velocity, slope = float(v_key[i]), float(s_key[j])
                            fea_save = np.zeros((2+6+8+24,))
                            knee_fea = imp_cal.cal_gait_fea()
                            knee_fea[4:] = knee_fea[4:]/100
                            knee_fea[:4] = np.deg2rad(knee_fea[:4])
                            thigh_fea = cal_gait_fea_thigh(q_t)
                            thigh_fea[:3] = np.deg2rad(thigh_fea[:3])
                            thigh_fea[3:] = thigh_fea[3:]/100
                            gait_fea = np.zeros((6+8,))
                            gait_fea[0:6] = thigh_fea[:]
                            gait_fea[6:] = knee_fea[:] 
                            fea_save[0:2] = [velocity, slope]
                            fea_save[2:16] = gait_fea[:]
                            fea_save[16:24] = (np.array(k_mat)*30).flatten()
                            fea_save[24:32] = (np.array(b_mat)*30).flatten()
                            fea_save[32:] = np.array(q_e_mat).flatten()
                        else:
                            print("========False========")        
                        if len(w) == 0 and avaliable:
                            imp_dataset.append(fea_save)
                        else:
                            print(w.message)
                        # ax1.cla()
                        # ax2.cla()
                except Exception as e:
                    # ax1.cla()
                    # ax2.cla()
                    pass
print(len(imp_dataset))
np.save("./result/predictor_dataset.npy", np.array(imp_dataset))  