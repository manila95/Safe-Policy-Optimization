import os
import pickle
import torch
import numpy as np
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import tqdm

def train_risk(model, dataloader, criterion, opt, num_epochs, device):
    model.train()
    net_loss = 0
    for _ in tqdm.tqdm(range(num_epochs)):
        for batch in dataloader:
                pred = model(batch[0].to(device))
                loss = criterion(pred, torch.argmax(batch[1].squeeze(), axis=1).to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                net_loss += loss.item()
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "risk_model.pt"))
    wandb.save("risk_model.pt")
    model.eval()
    return net_loss





def make_dirs(traj_path, episode):
        #try:
        os.makedirs(os.path.join(traj_path, "traj_%d"%episode, "lidar"))
        os.makedirs(os.path.join(traj_path, "traj_%d"%episode, "info"))
        
        #except:
        #    pass


def compute_fear(costs, max_dist=1000):
        fear_fwd, fear_bwd = torch.full(costs.size(), max_dist), torch.full(costs.size(), max_dist)
        fwd_flag, bwd_flag = 0, 0
        fwd_counter, bwd_counter = 0, 0
        len_run = len(costs)
        for i in range(len_run):
                if costs[i] == 1:
                        fear_fwd[i] = 0
                        fwd_flag = 1
                        fwd_counter = 0
                elif fwd_flag:
                       fwd_counter += 1
                       fear_fwd[i] = fwd_counter

                if costs[len_run-i-1] == 1:
                        bwd_flag = 1
                        fear_bwd[len_run-i-1] = 0
                        bwd_counter = 0
                elif bwd_flag:
                       bwd_counter += 1
                       fear_bwd[len_run-i-1] = bwd_counter
        return torch.min(fear_fwd, fear_bwd)

                     


def store_data(next_obs, info_dict, traj_path, episode, step_log):
        #, 'prev_obs_rgb': obs['vision']}
        #info_dict.update(obs)
        ## Saving the info for this step
        f1 = open(os.path.join(traj_path, "traj_%d"%episode, "info", "%d.pkl"%step_log), "wb")
        pickle.dump(info_dict, f1, protocol=pickle.HIGHEST_PROTOCOL)
        f1.close()
        # del obs['vision']
        ## Saving data from other sensors (particularly lidar)
        f2 = open(os.path.join(traj_path, "traj_%d"%episode, "lidar", "%d.pkl"%step_log), "wb")
        pickle.dump(next_obs, f2, protocol=pickle.HIGHEST_PROTOCOL)
        f2.close()


def get_activation(name):
    activation_dict = {
        'relu': nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
        "logsoftmax": nn.LogSoftmax(dim=1),
    }

    return activation_dict[name]



def make_state_action_risk_data(data_path):
        obs = torch.load(os.path.join(data_path, "obs.pt"))
        actions = torch.load(os.path.join(data_path, "actions.pt"))
        risks = torch.load(os.path.join(data_path, "risks.pt"))
        ep_len = torch.load(os.path.join(data_path, "ep_len.pt"))
        state_action_risk_data = None
        for idx in range(1, len(ep_len)):
                start, end = int(ep_len[idx-1]), int(ep_len[idx])
                print(start, end)
                obs_idx = obs[start:end]
                actions_idx = actions[start:end]
                risks_idx = risks[start:end]
                print(obs_idx.size(), actions_idx.size(), risks_idx.size())
                sar_data = torch.cat([obs_idx[:-1], actions_idx[1:], risks_idx[1:]], axis=1)
                state_action_risk_data = sar_data if state_action_risk_data is None else torch.cat([state_action_risk_data, sar_data], axis=0)
        torch.save(state_action_risk_data, os.path.join(data_path, "state_action_risk.pt"))
        return state_action_risk_data

def make_state_risk_data(data_path):
        obs = torch.load(os.path.join(data_path, "obs.pt"))
        risks = torch.load(os.path.join(data_path, "risks.pt"))
        ep_len = torch.load(os.path.join(data_path, "ep_len.pt"))
        return torch.cat([obs, risks], axis=1)

def combine_data(data_path, type="state_risk"):
        for env in os.listdir(data_path):
                env_path = os.path.join(data_path, env)
                all_data = None
                for run in os.listdir(env_path):
                        run_path = os.path.join(env_path, run)
                        if type == "state_risk":
                                try:
                                        data = make_state_risk_data(run_path)
                                except:
                                        pass
                        else:
                                try:
                                        data = make_state_action_risk_data(run_path)
                                except:
                                        pass
                all_data = data if all_data is None else torch.cat([all_data, data], axis=0)
        torch.save(all_data, os.path.join(env_path, "all_%s.pt"%type))



class ReplayBuffer:
        def __init__(self, buffer_size, obs_dim, risk_size, device):
                self.obs = None
                self.next_obs = torch.zeros(buffer_size, obs_dim).to(device)
                self.actions = None
                self.rewards = None
                self.dones = None
                self.risks = torch.zeros(buffer_size, risk_size).to(device)
                self.dist_to_fails = torch.zeros(buffer_size, 1).to(device)
                self.costs = None
                #self.data_path = data_path
                self.buffer_size = buffer_size
                self.buff_fill = 0

        def add(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                data_size = next_obs.size()[0]
                self.next_obs[self.buff_fill:self.buff_fill+data_size, :] = next_obs.squeeze()
                self.risks[self.buff_fill:self.buff_fill+data_size, :] = risk.squeeze()
                self.dist_to_fails[self.buff_fill:self.buff_fill+data_size, :] = dist_to_fail.reshape(-1, 1)
                self.buff_fill += data_size

                #self.obs = obs if self.obs is None else torch.concat([self.obs, obs], axis=0)
                #self.next_obs = next_obs if self.next_obs is None else torch.concat([self.next_obs, next_obs], axis=0)
                #self.actions = action if self.actions is None else torch.concat([self.actions, action], axis=0)
                #self.rewards = reward if self.rewards is None else torch.concat([self.rewards, reward], axis=0)
                #self.dones = done if self.dones is None else torch.concat([self.dones, done], axis=0)
                #self.risks = risk if self.risks is None else torch.concat([self.risks, risk], axis=0)
                #self.costs = cost if self.costs is None else torch.concat([self.costs, cost], axis=0)
                #self.dist_to_fails = dist_to_fail if self.dist_to_fails is None else torch.concat([self.dist_to_fails, dist_to_fail], axis=0)

        def __len__(self):
            return self.buff_fill

        def sample(self, sample_size):
                #if self.next_obs.size()[0] > self.buffer_size:
                #    self.next_obs = self.next_obs[-self.buffer_size:]
                #    self.risks = self.risks[-self.buffer_size:]
                sample_idx = np.random.randint(1, self.buff_fill, size=sample_size)
                return {"obs": None, #self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": None, #self.actions[sample_idx],
                        "rewards": None, #self.rewards[sample_idx],
                        "dones": None, #self.dones[sample_idx],
                        "risks": self.risks[sample_idx],
                        "costs": None, #self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}
        
        def sample_balanced(self, sample_size):
                idx = range(self.obs.size()[0])
                print(self.risks.size())
                
                idx_risky = idx[torch.argmax(self.risks, 1).squeeze().cpu().numpy() == 1]
                idx_safe  = idx[torch.argmax(self.risks, 1).squeeze().cpu().numpy() == 0]
                sample_idx = np.array(list(np.random.choice(idx_risky, sample_size/2)) + list(np.random.choice(idx_safe, sample_size/2)))
                return {"obs": self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": self.actions[sample_idx],
                        "rewards": self.rewards[sample_idx],
                        "dones": self.dones[sample_idx],
                        "risks": self.risks[sample_idx], 
                        "costs": self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}
                  

        def slice_data(self, min_idx, max_idx):
                idx = range(min_idx, max_idx)
                sample_idx = idx #np.random.choice(idx, sample_size)
                return {"obs": self.obs[sample_idx],
                        "next_obs": self.next_obs[sample_idx],
                        "actions": self.actions[sample_idx],
                        "rewards": self.rewards[sample_idx],
                        "dones": self.dones[sample_idx],
                        "risks": self.risks[sample_idx], 
                        "costs": self.costs[sample_idx],
                        "dist_to_fail": self.dist_to_fails[sample_idx]}        

        def save(self):
            torch.save(self.next_obs, os.path.join(self.data_path, "all_obs.pt"))
            torch.save(self.risks, os.path.join(self.data_path, "all_risks.pt"))



class ReplayBufferBalanced:
        def __init__(self, buffer_size=100000):
                self.obs_risky = None 
                self.next_obs_risky = None
                self.actions_risky = None 
                self.rewards_risky = None 
                self.dones_risky = None
                self.risks_risky = None 
                self.dist_to_fails_risky = None 
                self.costs_risky = None

                self.obs_safe = None 
                self.next_obs_safe = None
                self.actions_safe = None 
                self.rewards_safe = None 
                self.dones_safe = None
                self.risks_safe = None 
                self.dist_to_fails_safe = None 
                self.costs_safe = None

        def add_risky(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                self.obs_risky = obs if self.obs_risky is None else torch.concat([self.obs_risky, obs], axis=0)
                self.next_obs_risky = next_obs if self.next_obs_risky is None else torch.concat([self.next_obs_risky, next_obs], axis=0)
                self.actions_risky = action if self.actions_risky is None else torch.concat([self.actions_risky, action], axis=0)
                self.rewards_risky = reward if self.rewards_risky is None else torch.concat([self.rewards_risky, reward], axis=0)
                self.dones_risky = done if self.dones_risky is None else torch.concat([self.dones_risky, done], axis=0)
                self.risks_risky = risk if self.risks_risky is None else torch.concat([self.risks_risky, risk], axis=0)
                self.costs_risky = cost if self.costs_risky is None else torch.concat([self.costs_risky, cost], axis=0)
                self.dist_to_fails_risky = dist_to_fail if self.dist_to_fails_risky is None else torch.concat([self.dist_to_fails_risky, dist_to_fail], axis=0)

        def add_safe(self, obs, next_obs, action, reward, done, cost, risk, dist_to_fail):
                self.obs_safe = obs if self.obs_safe is None else torch.concat([self.obs_safe, obs], axis=0)
                self.next_obs_safe = next_obs if self.next_obs_safe is None else torch.concat([self.next_obs_safe, next_obs], axis=0)
                self.actions_safe = action if self.actions_safe is None else torch.concat([self.actions_safe, action], axis=0)
                self.rewards_safe = reward if self.rewards_safe is None else torch.concat([self.rewards_safe, reward], axis=0)
                self.dones_safe = done if self.dones_safe is None else torch.concat([self.dones_safe, done], axis=0)
                self.risks_safe = risk if self.risks_safe is None else torch.concat([self.risks_safe, risk], axis=0)
                self.costs_safe = cost if self.costs_safe is None else torch.concat([self.costs_safe, cost], axis=0)
                self.dist_to_fails_safe = dist_to_fail if self.dist_to_fails_safe is None else torch.concat([self.dist_to_fails_safe, dist_to_fail], axis=0)

        
        def sample(self, sample_size):
                idx_risky = range(self.obs_risky.size()[0])
                idx_safe = range(self.obs_safe.size()[0])

                sample_risky_idx = np.random.choice(idx_risky, int(sample_size/2))
                sample_safe_idx = np.random.choice(idx_safe, int(sample_size/2))

                return {"obs": torch.cat([self.obs_risky[sample_risky_idx], self.obs_safe[sample_safe_idx]], 0),
                        "next_obs": torch.cat([self.next_obs_risky[sample_risky_idx], self.next_obs_safe[sample_safe_idx]], 0),
                        "actions": torch.cat([self.actions_risky[sample_risky_idx], self.actions_safe[sample_safe_idx]], 0),
                        "rewards": torch.cat([self.rewards_risky[sample_risky_idx], self.rewards_safe[sample_safe_idx]], 0),
                        "dones": torch.cat([self.dones_risky[sample_risky_idx], self.dones_safe[sample_safe_idx]], 0),
                        "risks": torch.cat([self.risks_risky[sample_risky_idx], self.risks_safe[sample_safe_idx]], 0),
                        "costs": torch.cat([self.costs_risky[sample_risky_idx], self.costs_safe[sample_safe_idx]], 0),
                        "dist_to_fail": torch.cat([self.dist_to_fails_risky[sample_risky_idx], self.dist_to_fails_safe[sample_safe_idx]], 0),}
        



                        

                


class BayesRiskEstCont(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128, fc3_size=128, fc4_size=128, out_size=1, model_type="state_risk", action_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.model_type = model_type
        self.action_size = action_size

        self.fc1 = nn.Linear(obs_size, fc1_size)
        if self.model_type == "state_risk":
            self.fc2 = nn.Linear(fc1_size, fc2_size)
        else:
            self.fc1_action = nn.Linear(action_size, int(fc1_size/2))
            self.fc2 = nn.Linear(fc1_size + int(fc1_size/2), fc2_size)
            self.bnorm1_action = nn.BatchNorm1d(int(fc1_size/2))

        self.mean_fc3 = nn.Linear(fc2_size, fc3_size)
        self.mean_fc4 = nn.Linear(fc3_size, fc4_size)
        self.mean_out = nn.Linear(fc4_size, out_size)

        self.logvar_fc3 = nn.Linear(fc2_size, fc3_size)
        self.logvar_fc4 = nn.Linear(fc3_size, fc4_size)
        self.logvar_out = nn.Linear(fc4_size, out_size)


        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.mean_bnorm3 = nn.BatchNorm1d(fc3_size)
        self.mean_bnorm4 = nn.BatchNorm1d(fc4_size)

        #self.var_bnorm1 = nn.BatchNorm1d(fc1_size)
        #self.var_bnorm2 = nn.BatchNorm1d(fc2_size)
        self.var_bnorm3 = nn.BatchNorm1d(fc3_size)
        self.var_bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, action=None):
        x = self.bnorm1(self.relu(self.fc1(x)))
        if self.model_type == "state_action_risk":
            x1 = self.bnorm1_action(self.relu(self.fc1_action(action)))
            x = torch.cat([x, x1], axis=1)

        x = self.bnorm2(self.relu(self.fc2(x)))

        mean  = self.mean_bnorm3(self.relu(self.mean_fc3(x)))
        mean  = self.mean_bnorm4(self.relu(self.mean_fc4(mean)))
        mean  = self.sigmoid(self.mean_out(mean))

        logvar = self.var_bnorm3(self.relu(self.logvar_fc3(x)))
        logvar = self.var_bnorm4(self.relu(self.logvar_fc4(x)))
        logvar = self.sigmoid(self.logvar_out(x))

        #x = self.bnorm3(self.relu(self.dropout(self.fc3(x))))
        #x = self.bnorm4(self.relu(self.dropout(self.fc4(x))))
        #out = self.logsoftmax(self.out(x))
        return mean, logvar



class BayesRiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=64, fc2_size=64,\
                  fc3_size=64, fc4_size=64, out_size=2, batch_norm=True, activation='relu', model_type="state_risk", action_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
        self.model_type = model_type
        self.fc1 = nn.Linear(obs_size, fc1_size)
        if self.model_type == "state_risk":
            self.fc2 = nn.Linear(fc1_size, fc2_size)
        else:
            self.fc1_action = nn.Linear(action_size, int(fc1_size/2))
            self.fc2 = nn.Linear(fc1_size + int(fc1_size/2), fc2_size)
            self.bnorm1_action = nn.BatchNorm1d(int(fc1_size/2))

        #self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)

        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.bnorm3 = nn.BatchNorm1d(fc3_size)
        self.bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.activation = get_activation(activation)

        self.logsoftmax = get_activation("logsoftmax")
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, action=None):
        # Taking care of any augmentation in the observation space
        x = x[:, :self.obs_size]
        if self.batch_norm:
            x = self.bnorm1(self.activation(self.fc1(x)))
            if self.model_type == "state_action_risk":
                x1 = self.bnorm1_action(self.activation(self.fc1_action(action)))
                x = torch.cat([x, x1], axis=1)
            #x = self.bnorm2(self.activation(self.fc2(x)))
            # x = self.bnorm3(self.activation(self.dropout(self.fc3(x))))
            x = self.bnorm4(self.activation(self.dropout(self.fc4(x))))
        else:
            x = self.activation(self.fc1(x))
            if self.model_type == "state_action_risk":
                x1 = self.activation(self.fc1_action(action))
                x = torch.cat([x, x1], axis=1)

            #x = self.activation(self.fc2(x))
            # x = self.activation(self.dropout(self.fc3(x)))
            x = self.activation(self.dropout(self.fc4(x)))

        out = self.logsoftmax(self.out(x))
        return out


class RiskEst(nn.Module):
    def __init__(self, obs_size=64, fc1_size=128, fc2_size=128,\
                  fc3_size=128, fc4_size=128, out_size=2, batch_norm=False, activation='relu', continuous_risk=False):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
        self.continuous_risk = continuous_risk

        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        self.out = nn.Linear(fc4_size, out_size)

        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)
        self.bnorm3 = nn.BatchNorm1d(fc3_size)
        self.bnorm4 = nn.BatchNorm1d(fc4_size)

        # Activation functions
        self.activation = get_activation(activation)
        self.softmax = get_activation("softmax")

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if self.batch_norm:
            x = self.bnorm1(self.activation(self.fc1(x)))
            x = self.bnorm2(self.activation(self.fc2(x)))
            x = self.bnorm3(self.activation(self.dropout(self.fc3(x))))
            x = self.bnorm4(self.activation(self.dropout(self.fc4(x))))
        else:
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.dropout(self.fc3(x)))
            x = self.activation(self.dropout(self.fc4(x)))    
        
        if self.continuous_risk:
            out = self.sigmoid(self.out(x))
        else:
            out = self.softmax(self.out(x))
        return out