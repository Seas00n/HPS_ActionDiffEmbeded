import numpy as np

exp_dataset = np.load("./result/exp_dataset.npz")

cal_dataset = np.load("./result/predictor_dataset.npy")


A_exp = exp_dataset["real_imp"]
A_max = np.max(A_exp, axis=0)
A_min = np.min(A_exp, axis=0)
nor_A_exp = (A_exp-A_min)/(A_max-A_min)
nor_A_exp = np.clip(nor_A_exp, a_min=0, a_max=1)

A_cal = cal_dataset[:,16:]
nor_A_cal = (A_cal-A_min)/(A_max-A_min)
nor_A_cal = np.clip(nor_A_cal, a_min=0, a_max=1)


fea_k_exp = exp_dataset['gaitfea_k'][:,0,:]
fea_t_exp = exp_dataset['gaitfea_t'][:,0,:]
fea_k_exp = fea_k_exp[:,[2,3,6,7]]
fea_k_exp[:,-1] = 0.99
fea_t_exp = fea_t_exp[:,[0,1,2,4]]

X_exp = np.hstack([fea_t_exp, fea_k_exp])
X_max = np.max(X_exp, axis=0)
X_min = np.min(X_exp, axis=0)
nor_X_exp = np.zeros_like(X_exp)
for i in range(X_exp.shape[1]):
    if X_min[i] == X_max[i]:
        nor_X_exp[:,i] = X_min[i]
    else:
        nor_X_exp[:,i] = (X_exp[:,i]-X_min[i])/(X_max[i]-X_min[i])
nor_X_exp = np.clip(nor_X_exp,0,1)

X_cal = cal_dataset[:,[2,3,4,6,10,11,14,15]]
nor_X_cal = np.zeros_like(X_cal)
for i in range(X_exp.shape[1]):
    if X_min[i] == X_max[i]:
        nor_X_cal[:,i] = X_min[i]
    else:
        nor_X_cal[:,i] = (X_cal[:,i]-X_min[i])/(X_max[i]-X_min[i])
nor_X_cal = np.clip(nor_X_cal, 0,1)


lim_v = [0.6, 1.2]    

lim_s = [-10., 10.]

min_value = np.array([lim_v[0], lim_s[0]])
max_value = np.array([lim_v[1], lim_s[1]])


C_exp = exp_dataset['context']

C_exp = C_exp*(max_value-min_value)+min_value
C_exp = np.round(np.clip(C_exp, min_value, max_value),2)

C_cal = cal_dataset[:,:2]


np.savez("./result/exp_dataset_nor.npz", 
         context=C_exp,
         state=nor_X_exp,
         action=nor_A_exp)

np.savez("./result/cal_dataset_nor.npz",
         context=C_cal,
         state=nor_X_cal,
         action=nor_A_cal)

print("Save New Dataset")