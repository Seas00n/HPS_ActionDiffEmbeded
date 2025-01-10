import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import argrelextrema
from scipy.interpolate import griddata

v_s_t_grid = np.load("v_s_t_grid.npy")

idx = np.arange(0,101)
m = 5
L = 0.5
r = 0.04
# iner_1 = m*(0.5*L)**2+1/12*(3*r**2+L**2)*m
# iner_2 = m*9.8*L/2
# iner_3 = 1

iner_1 = 0.06200995
iner_2 = 4.33530566
iner_3 = 1

def cal_mid_k(v1, v2):
    theta1 = np.arctan2(v1[1], v1[0])
    theta2 = np.arctan2(v2[1], v2[0])
    return np.tan((theta1+theta2)/2)

def grid_interp_dt(v, s, num_frame):
    v = np.clip(v, v_s_t_grid[0,0,0], v_s_t_grid[-1,0,0])
    dv = v_s_t_grid[1,0,0]-v_s_t_grid[0,0,0]
    s = np.clip(s, v_s_t_grid[0,0,1], v_s_t_grid[0,-1,1])
    ds = v_s_t_grid[0,1,1]-v_s_t_grid[0,0,1]
    idx_v = int((v-v_s_t_grid[0,0,0])/dv)
    idx_s = int((s-v_s_t_grid[0,0,1])/ds)
    if idx_v == v_s_t_grid.shape[0]-1:
        idx_v -= 1
    if idx_s == v_s_t_grid.shape[1]-1:
        idx_s -= 1

    v_1, v_2 = v_s_t_grid[idx_v, 0, 0], v_s_t_grid[idx_v+1,0,0]
    k_v = (v-v_1)/(v_2-v_1)
    s_1, s_2 = v_s_t_grid[0, idx_s, 1], v_s_t_grid[0,idx_s+1,1]
    k_s = (s-s_1)/(s_2-s_1)
    
    t_vmid_s1 = k_v*(v_s_t_grid[idx_v+1, idx_s, 2]-v_s_t_grid[idx_v, idx_s, 2])+v_s_t_grid[idx_v, idx_s, 2]
    t_vmid_s2 = k_v*(v_s_t_grid[idx_v+1, idx_s+1, 2]-v_s_t_grid[idx_v, idx_s+1, 2])+v_s_t_grid[idx_v, idx_s+1, 2]
    t = k_s*(t_vmid_s2-t_vmid_s1)+t_vmid_s1
    return t/(num_frame-1)


def cal_qe_knee_phase1(q_k, t_k):
    p_1 = np.argmax(t_k[:25])
    p_0 = np.argmin(t_k[:p_1])
    if p_0 < 0:
        p_0 = 0
    p_2 = np.argmin(t_k[30:40])+30
    imp_k = [0,0]
    def staticimpFun(x, k, qe):
        return k*(x-qe)
    # va = np.array([q_k[p_2]-q_k[p_1], t_k[p_2]-t_k[p_1]])
    # va = va/np.linalg.norm(va)
    # vb = np.array([q_k[p_0]-q_k[p_1], t_k[p_0]-t_k[p_1]])
    # vb = vb/np.linalg.norm(vb)
    # angle = np.rad2deg(np.arcsin(np.cross(va, vb)))
    imp_k,pcov = curve_fit(staticimpFun, q_k[p_0:p_1],
                        t_k[p_0:p_1], bounds=((0,0),(100,40)))
    return imp_k, p_0, p_1, p_2

def cal_qe_knee_phase2(q_k, t_k):
    idx_max_t_k = np.argmax(t_k[10:30])+10
    local_min_list = argrelextrema(t_k[idx_max_t_k:60], np.less)[0]
    if len(local_min_list)>1:
        print(local_min_list)
        for i in local_min_list:
            if i > 10 and i < 30:
                p_1 = i+idx_max_t_k
    else:
        p_1 =local_min_list[0] +idx_max_t_k
    p_0 = p_1 - 10
    p_2 = p_1 + 10
    imp_k = [0,0]
    def staticimpFun(x, k, qe):
        return k*(x-qe)
    # imp_k[0] = cal_mid_k(
    #     v1=[q_k[p_0]-q_k[p_1], t_k[p_0]-t_k[p_1]],
    #     v2=[q_k[p_2]-q_k[p_1], t_k[p_2]-t_k[p_1]])
    # imp_k[1] = q_k[p_1]-t_k[p_1]/imp_k[0]
    imp_k,_ = curve_fit(staticimpFun, q_k[p_1:p_2],
                        t_k[p_1:p_2], bounds=((0,0),(5,30)))
    return imp_k, p_0, p_1, p_2

def cal_qe_ankle_phase1(q_a, t_a):
    p_1 = np.argmin(t_a[0:15])
    # p_0 = p_1
    p_2 = p_1 + 15
    p_1 = p_2 - 5
    p_0 = p_1
    def staticimpFun(x, k, qe):
        return k*(x-qe)
    imp_a,_ = curve_fit(staticimpFun, q_a[p_1:p_2],
                        t_a[p_1:p_2], bounds=((0,-15),(7,15)))
    return imp_a, p_0, p_1, p_2

def cal_qe_ankle_phase2(q_a, t_a):
    p_1 = np.argmax(t_a)
    p_0 = p_1 - 10
    idx_max_t_a = np.argmax(t_a)
    p_2 = np.argmin(q_a[idx_max_t_a:])+idx_max_t_a
    max_qa = np.max(q_a[40:80])
    def staticimpFun(x, k, qe):
        return k*(x-qe)
    imp_a1,_ = curve_fit(staticimpFun, q_a[p_1:p_2],
                        t_a[p_1:p_2], bounds=((0,-20),(15,15)))
    # imp_a2,pcov = curve_fit(staticimpFun, q_a[p_0:p_1],
    #                      t_a[p_0:p_1], bounds=((0,-5),(40,max_qa)))
    imp_a = imp_a1
    return imp_a, p_0, p_1, p_2

def cal_tpred(q, dq, imp,reverse=False):
    if not reverse:
        return imp[0]*(q-imp[2])+imp[1]*dq
    else:
        return imp[0]*(q+imp[2])+imp[1]*dq

def cal_imp_kbqe(q, dq, t, bound, unbound=False, reverse=False):
    if not reverse:
        def impFun(x, k, b, qe):
            return k*(x[0]-qe)+(b+0.0005)*x[1]
    else:
        def impFun(x, k, b, qe):
            return k*(x[0]+qe)+(b+0.0005)*x[1]
    if unbound:
        imp,_ = curve_fit(impFun,[q, dq],t,bounds=((0,0,0),(10,1,90)))
    else:
        imp,_ = curve_fit(impFun,[q, dq],t,bounds=bound)
    t_pred = impFun([q,dq], *imp)
    return imp, t_pred

def cal_imp_kb(q, dq, qe, t, bound, unbound=False, reverse=False):
    if not reverse:
        def impFun(x, k, b):
            return k*(x[0]-qe)+b*x[1]
    else:
        def impFun(x, k, b):
            return k*(x[0]+qe)+b*x[1]
    if unbound:
        imp, _ = curve_fit(impFun, [q, dq], t)
    else:
        imp, _ = curve_fit(impFun, [q, dq], t, bounds=bound)
    t_pred = impFun([q,dq], *imp)
    imp = np.array([imp[0], imp[1], qe])
    return imp, t_pred

def cal_imp_bqe(q, dq, k, t, bound, unbound=False, reverse=False):
    if not reverse:
        def impFun(x, b, qe):
            return k*(x[0]-qe)+b*x[1]
    else:
        def impFun(x, b, qe):
            return k*(x[0]+qe)+b*x[1]
    if unbound:
        imp, _ = curve_fit(impFun, [q, dq], t)
    else:
        imp, _ = curve_fit(impFun, [q, dq], t, bounds=bound)
    t_pred = impFun([q,dq], *imp)
    imp = np.array([k, imp[0], imp[1]])
    return imp, t_pred

def get_parabola(q0, s1, q1, s2, q2, dt=0.01):
    q_new = np.zeros((s2+1,))
    a1 = (q0-q1)/s1**2
    b1, c1 = -2*s1*a1, q0
    x1 = np.arange(0,s1+1)
    q_new[0:s1+1] = a1*x1**2+b1*x1+c1
    a2 = (q2-q1)/(s1-s2)**2
    b2, c2 = -2*a2*s1, q1+a2*s1**2
    x2 = np.arange(s1+1, s2+1)
    q_new[s1+1:s2+1] = a2*x2**2+b2*x2+c2
    t_new = np.arange(s2+1)*dt
    return q_new, t_new

def system_model(x,dx,u,dt,iner=[iner_1,iner_2,iner_3]):#[0.06200995,4.33530566,1]
    ddx = (u-iner[2]*dx-iner[1]*np.sin(x))/iner[0]
    dx = ddx*dt+dx
    x = dx*dt+x
    return x, dx