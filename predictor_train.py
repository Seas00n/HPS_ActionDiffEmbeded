import numpy as np
import torch
import torch.nn as nn   
import torch.optim as optim
import torch.nn.functional as F 
from collections import defaultdict
from algo import ActChangeNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

dataset = np.load("./result/cal_dataset_nor.npz")
state = dataset['state']
context = dataset['context']
action = dataset['action']

class ActionChangeDataset:
    def __init__(self, states, contexts, actions):
        """Dataset for (state, context, action) tuples.
        Args:
            states (np.ndarray): Array of states.
            contexts (np.ndarray): Array of contexts.
            actions (np.ndarray): Array of actions.
        """
        self.states = torch.tensor(states, dtype=torch.float32).to(device)
        self.contexts = torch.tensor(contexts, dtype=torch.float32).to(device)
        self.actions = torch.tensor(actions, dtype=torch.float32).to(device)
        self.num_data = len(self.states)
        self.context_groups = defaultdict(list)
        for idx, context in enumerate(self.contexts):
            self.context_groups[str(tuple(context.cpu().numpy()))].append(idx)
        self.context_keys = list(self.context_groups.keys())
        print()
    
    def get_context_keys(self):
        """
        Get all unique context keys.

        Returns:
            List[tuple]: List of unique context keys.
        """
        return self.context_keys
    
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

        # state_diffs = self.states[sampled_indices_i] - self.states[sampled_indices_j]
        # action_diffs = self.actions[sampled_indices_i] - self.actions[sampled_indices_j]
        selected_contexts = torch.asarray(np.array(selected_contexts)).to(device)
        state_diffs_interp = torch.vstack([(self.states[sampled_indices_i]-self.states[sampled_indices_j])*alpha for alpha in np.linspace(0,1,3)])
        action_diffs_interp = torch.vstack([(self.actions[sampled_indices_i]-self.actions[sampled_indices_j])*alpha for alpha in np.linspace(0,1,3)])
        contexts_noise = torch.randn(selected_contexts.shape).to(device)
        contexts_noise[:,0] = torch.clip(contexts_noise[:,0],-1,1)*0.2
        contexts_noise[:,1] = torch.clip(contexts_noise[:,1],-1,1)*2.5
        selected_contexts = selected_contexts+contexts_noise
        selected_contexts_interp = torch.vstack([selected_contexts for i in range(3)])
        return state_diffs_interp, action_diffs_interp, selected_contexts_interp


diffset = ActionChangeDataset(states=state, contexts=context, actions=action)


delta_act_model = ActChangeNN(state_dim=state.shape[1], context_dim=context.shape[1], action_dim=action.shape[1]).to(device)
optimizer_action = optim.Adam(delta_act_model.parameters(), lr=1e-4)

num_epoch_action = 200000
batch_size = 30
for epoch in range(num_epoch_action):
    total_loss = 0.0
    state_diff, act_diff, context_selected = diffset.sample_random_pairs(batch_size)

    predicted_actdiff = delta_act_model(state=state_diff, context=context_selected)
    loss = F.mse_loss(predicted_actdiff, act_diff)

    optimizer_action.zero_grad()
    loss.backward()
    optimizer_action.step()
    if (epoch+1)%5000 == 0:
        print("Itr"+str(epoch+1)+" Training Loss"+"{:4}".format(loss.cpu().data.numpy()))

torch.save(delta_act_model.state_dict(), "./result/act_diff_{}.pth".format(num_epoch_action))