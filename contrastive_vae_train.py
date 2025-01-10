import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from algo import ActChangeNN, VAE, LatentDynNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = np.load("./result/exp_dataset_nor.npz")
state = dataset['state']
context = dataset['context']
action  = dataset['action']

actdiffnn = ActChangeNN(state_dim=state.shape[1],context_dim=context.shape[1],action_dim=action.shape[1]).to(device)
actdiffnn.load_state_dict(torch.load("./result/act_diff_200000.pth", map_location=device))
for param in actdiffnn.parameters():
    param.requires_grad = False

class ActChangeDataset:
    def __init__(self, states, contexts, actions):
        self.states = torch.tensor(states, dtype=torch.float32).to(device)
        self.contexts = torch.tensor(contexts, dtype=torch.float32).to(device)
        self.actions = torch.tensor(actions, dtype=torch.float32).to(device)
        self.num_data = len(self.states)
        self.context_groups = defaultdict(list)
        for idx, context in enumerate(self.contexts):
            self.context_groups[str(tuple(context.cpu().numpy()))].append(idx)
        self.context_keys = list(self.context_groups.keys())
        print(self.context_keys)
    
    def sample_random_pairs(self, num_pairs):
        selected_context_indices = np.random.randint(0, len(self.context_keys), (num_pairs,))
        sampled_indices_i = []
        sampled_indices_j = []
        selected_contexts = []
        for context_idx in selected_context_indices:
            indices = self.context_groups[self.context_keys[context_idx]]
            if len(indices) < 2:
                raise ValueError(f"Not enough samples for context {self.context_keys[context_idx]}")

            # Sample two random indices from this context
            pair_indices = np.random.randint(0, len(indices), (2,))
            sampled_indices_i.append(indices[pair_indices[0]])
            sampled_indices_j.append(indices[pair_indices[1]])
            context_str = self.context_keys[context_idx]
            selected_contexts.append(np.array(context_str.strip("()").split(", ")).astype(np.float32))
        
        selected_contexts = torch.asarray(np.array(selected_contexts)).to(device)
        state_diffs_interp = torch.vstack([(self.states[sampled_indices_i]-self.states[sampled_indices_j])*alpha for alpha in np.linspace(0,1,5)[:3]]).to(device)
        context_interp = torch.vstack([selected_contexts for i in range(3)]).to(device)
        action_pred_interp = torch.vstack([self.actions[sampled_indices_i] for i in range(3)]).to(device)
        action_diff_interp = torch.zeros_like(action_pred_interp).to(device)
        for i in range(3):
            idx_current = torch.arange(num_pairs)+i*num_pairs
            act_diff = actdiffnn(state=state_diffs_interp[idx_current], context=context_interp[idx_current])
            if i == 0:
                action_pred_interp[idx_current] = torch.clone(self.actions[sampled_indices_i])
            if i < 2:
                idx_next = torch.arange(num_pairs)+(i+1)*num_pairs
                action_pred_interp[idx_next] = action_pred_interp[idx_current]+act_diff
            action_diff_interp[idx_current] = act_diff
        idx_new = np.arange(state_diffs_interp.shape[0])
        idx_new = idx_new.reshape((-1,num_pairs))
        idx_new = idx_new.T.reshape((-1,1)).flatten()
        state_diffs_interp = state_diffs_interp[idx_new]
        context_interp = context_interp[idx_new]
        action_pred_interp = action_pred_interp[idx_new]
        action_diff_interp = action_diff_interp[idx_new]
        return state_diffs_interp, context_interp, action_pred_interp, action_diff_interp

dataset = ActChangeDataset(state, context, action)

vae = VAE(context_dim=context.shape[1], action_dim=action.shape[1], latent_dim=4).to(device)
latent_dynNN = LatentDynNN(state_dim=state.shape[1], context_dim=context.shape[1], latent_dim=4).to(device)

optimizer = optim.Adam(list(vae.parameters())+list(latent_dynNN.parameters()), lr=1e-4)

num_epoch_action = 500000

batch_size=100

beta_kl = 0.1
beta_dyn = 0.1

for epoch in range(num_epoch_action):
    s_diff, c, a, a_diff = dataset.sample_random_pairs(batch_size)

    a_recon, e_mean, e_logvar = vae(context=c, action=a)

    recon_loss = F.mse_loss(a_recon, a)

    KL_loss = -0.5*(1+e_logvar-e_mean.pow(2)-e_logvar.exp()).mean()
    
    delta_e_pred = latent_dynNN(state_diff=s_diff, context=c, z=e_mean)

    a_pred_recon = vae.decode(context=c, z=e_mean+delta_e_pred)

    pred_loss = F.mse_loss(a_pred_recon, a+a_diff)

    total_loss = recon_loss+beta_kl*KL_loss+beta_dyn*pred_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch+1)%500 == 0:
        print("Itr"+str(epoch+1)+"Training Loss"+"{:.4}".format(total_loss.cpu().data.numpy())+
              ",ReconLoss"+"{:.4}".format(recon_loss.cpu().data.numpy())+
              ",KLLoss"+"{:.4}".format(KL_loss.cpu().data.numpy())+
              ",dynLoss"+"{:.4}".format(pred_loss.cpu().data.numpy()))

torch.save(vae.state_dict(), "./result/vae_{}.pth".format(num_epoch_action))
torch.save(latent_dynNN.state_dict(), "./result/latent_dynNN_{}.pth".format(num_epoch_action))


