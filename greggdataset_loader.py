import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import argrelextrema
from scipy.interpolate import griddata, interp1d
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, lfilter
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)


# 设定采样频率和截止频率
fs = 1000  # 采样频率，单位Hz
cutoff = 70  # 截止频率，单位Hz

# 设计一个巴特沃斯低通滤波器
order = 2  # 滤波器的阶数，阶数越高，滤波器的过渡带越窄
b, a = butter(order, 2 * cutoff / fs, btype='low')  # 正常化截止频率

def phase_shift(data, idx_shift):
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

path_ = current_dir+"/gregg_dataset/result/"

v_key = ["0.8","1.0","1.2"]
s_key = ["-10", "-5", "0", "5", "10"]  

def load_data_temp(ab_k, vk, sk):
    exists = (vk in v_key) and (sk in s_key)
    if not exists:
        print("Velocity or Slope not exists")
        print(v_key)
        print(s_key)
        return None
    if vk == "1.0": vk = "1"
    data = np.load(path_+ab_k+"/v{}_s{}.npy".format(vk, sk))
    mean_data = np.mean(data, axis=2)
    std_data = np.std(data, axis=2)
    dis_mean = data[4:6,:,:]-mean_data[4:6,:].reshape((2,-1,1))
    sig_mask = np.int64(abs(dis_mean)-std_data[4:6,:].reshape(2,-1,1)*2>0)
    sig_mask = np.sum(np.sum(sig_mask, axis=1),axis=0)
    idx_deleted = np.where(sig_mask>15)[0]
    data = np.delete(data, idx_deleted, axis=2)
    idx_deleted2 = np.where(np.max(-data[4,15:25,:]*60, axis=0)<5)[0]
    data = np.delete(data, idx_deleted2, axis=2)
    mean_data = np.mean(data, axis=2)
    return data, mean_data, std_data

def interp101(data, mean_data):
    data_new = np.zeros((data.shape[0],101,data.shape[2])) 
    x = np.linspace(0,100,150)
    xx = np.linspace(0,100,101)        
    fun = interp1d(x, data, axis=1)
    data_new = fun(xx)
    fun = interp1d(x, mean_data, axis=1)
    mean_data_new = fun(xx)
    return data_new, mean_data_new
    
def find_longest_consecutive_repeated_sequence(seq):
    if not seq :  # 如果序列为空，返回空序列
        return []

    max_length = 1
    current_length = 1
    longest_sequence_start = 0

    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                longest_sequence_start = i - max_length
            current_length = 1

    # 检查最后一个窗口
    if current_length > max_length:
        max_length = current_length
        longest_sequence_start = len(seq) - max_length

    return len(seq[longest_sequence_start:longest_sequence_start + max_length])

def load_data(vk, sk, ab_k="AB06",m=60,show=False, ax=None):
    exists = (vk in v_key) and (sk in s_key)
    if not exists:
        print("Velocity or Slope not exists")
        print(v_key)
        print(s_key)
        return None, None
    if vk == "1.0": vk = "1"
    ab_info = np.load(path_+ab_k+"/ab_info.npy")
    print(ab_info)
    data = np.load(path_+ab_k+"/v{}_s{}.npy".format(vk, sk))
    # data = data[:,:,5:-5]
    ######################################
    mean_data = np.mean(data, axis=2)
    std_data = np.std(data, axis=2)
    dis_mean = data[4:6,:,:]-mean_data[4:6,:].reshape((2,-1,1))
    sig_mask = np.int64(abs(dis_mean)-std_data[4:6,:].reshape(2,-1,1)*2>0)
    sig_mask = np.sum(np.sum(sig_mask, axis=1),axis=0)
    idx_deleted = np.where(sig_mask>15)[0]
    data = np.delete(data, idx_deleted, axis=2)
    #######################################
    equal_mask = np.zeros((data.shape[2],))
    for i in range(equal_mask.shape[0]):
        equal_mask[i] = np.int64(find_longest_consecutive_repeated_sequence((data[4,:,i]*m).tolist()))-1
        equal_mask[i] = equal_mask[i] + np.int64(find_longest_consecutive_repeated_sequence((data[5,:,i]*m).tolist()))-1
    idx_deleted = np.where(equal_mask>5)[0]
    data = np.delete(data, idx_deleted, axis=2)
    if show:
        ax.plot(np.arange(data.shape[1]), -data[4,:,:]*m, alpha=0.2, color='g')
    #######################################
    idx_deleted2 = np.where(np.max(-data[4,:25,:]*m, axis=0)<-5)[0]
    data = np.delete(data, idx_deleted2, axis=2)
    data = data[[1,4,2,5],:,:]
    data[0] = data[0]-2
    data[1] = -data[1]*m
    data[3] = -data[3]*m
    mean_data = np.mean(data, axis=2)
    data = lfilter(b, a, data, axis=1)
    data, mean_data = interp101(data, mean_data)
    if show:
        ax.plot(np.linspace(0,150,data.shape[1]), data[1,:,:], alpha=0.5, color='r')
    mean_data = gaussian_filter1d(mean_data, sigma=2, axis=1)
    return data, mean_data

def load_data_with_thigh(vk, sk, ab_k="AB06",m=60,show=False, ax=None):
    exists = (vk in v_key) and (sk in s_key)
    if not exists:
        print("Velocity or Slope not exists")
        print(v_key)
        print(s_key)
        return None, None
    if vk == "1.0": vk = "1"
    ab_info = np.load(path_+ab_k+"/ab_info.npy")
    print(ab_info)
    data = np.load(path_+ab_k+"/v{}_s{}.npy".format(vk, sk))
    # data = data[:,:,5:-5]
    ######################################
    mean_data = np.mean(data, axis=2)
    std_data = np.std(data, axis=2)
    dis_mean = data[4:6,:,:]-mean_data[4:6,:].reshape((2,-1,1))
    sig_mask = np.int64(abs(dis_mean)-std_data[4:6,:].reshape(2,-1,1)*2>0)
    sig_mask = np.sum(np.sum(sig_mask, axis=1),axis=0)
    idx_deleted = np.where(sig_mask>15)[0]
    data = np.delete(data, idx_deleted, axis=2)
    #######################################
    equal_mask = np.zeros((data.shape[2],))
    for i in range(equal_mask.shape[0]):
        equal_mask[i] = np.int64(find_longest_consecutive_repeated_sequence((data[4,:,i]*m).tolist()))-1
        equal_mask[i] = equal_mask[i] + np.int64(find_longest_consecutive_repeated_sequence((data[5,:,i]*m).tolist()))-1
    idx_deleted = np.where(equal_mask>5)[0]
    data = np.delete(data, idx_deleted, axis=2)
    if show:
        ax.plot(np.arange(data.shape[1]), -data[4,:,:]*m, alpha=0.2, color='g')
    #######################################
    idx_deleted2 = np.where(np.max(-data[4,:25,:]*m, axis=0)<-5)[0]
    data = np.delete(data, idx_deleted2, axis=2)
    data = data[[0,1,4,2,5],:,:]
    data[1] = data[1]-2
    data[2] = -data[2]*m
    data[4] = -data[4]*m
    mean_data = np.mean(data, axis=2)
    data[0,:,:] = gaussian_filter1d(data[0,:,:], sigma=2, axis=0)
    data[1:,:,:] = lfilter(b, a, data[1:,:,:], axis=1)
    data, mean_data = interp101(data, mean_data)
    if show:
        ax.plot(np.linspace(0,150,data.shape[1]), data[0,:,:], alpha=0.5, color='r')
    mean_data = gaussian_filter1d(mean_data, sigma=2, axis=1)
    return data, mean_data

# data, mean_data, std_data= load_data_temp("AB06","0.8","5")

# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)


# idx = np.arange(mean_data.shape[1])
# ax1.plot(idx, data[1,:,:], color='b', alpha=0.2)
# ax1.set_ylim(-1,80)
# ax2.plot(idx, data[2,:,:],color='b', alpha=0.2)
# ax2.set_ylim(-20,20)
# ax3.plot(idx, data[4,:,:], color='g', alpha=0.2)
# ax3.set_ylim(-1,1)
# # ax3.plot(idx, (-mean_data[4,:]-2*std_data[4,:])*60,'r--')
# # ax3.plot(idx, (-mean_data[4,:]+2*std_data[4,:])*60,'r--')

# ax4.plot(idx, data[5,:,:], color='g', alpha=0.2)
# ax4.set_ylim(-2, 0.5)


# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# idx_new = np.arange(101)
# data_new, mean_data_new = load_data(vk="0.8", sk="5", m=73)
# ax1.plot(mean_data_new[2,:], mean_data_new[3,:])

# ax2.plot(idx_new, data_new[1,:,:], color='b', alpha=0.2)
# ax2.plot(idx_new, data_new[1,:,:], color='g', alpha=0.2)

# # # ax4.plot(idx_new, data_new[2,:,:],color='b', alpha=0.2)
# # ax4.plot(idx_new, data_new[3,:,:], color='g', alpha=0.2)

# qa = mean_data_new[2,:]
# ta = mean_data_new[3,:]



# def calculate_curvature(x_value: np.ndarray, y_value: np.ndarray):
#     """计算曲率"""
#     x_t = np.gradient(x_value)
#     y_t = np.gradient(y_value)
#     xx_t = np.gradient(x_t)
#     yy_t = np.gradient(y_t)
#     curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** 1.5
#     return curvature_val



# idx = np.arange(101)
# cur = calculate_curvature(qa, ta)
# diff = np.gradient(ta)/np.gradient(qa)

# ax1.plot(idx, diff)
# ax2.plot(qa, ta)
# for i in range(50):
#     a = np.clip(diff[i], 0, 50)/50
#     if 20< diff[i] <40:
#         ax2.scatter(qa[i], ta[i], color='r', alpha=a)
#         print(diff[i])

# plt.show()