import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import tqdm
import numpy as np

from safepo.common.model import RiskEst



class RiskTrainer:
	def __init__(self, args, obs_size, risk_size, device):
		self.args = args
		self.device = device
		self.model = RiskEst(obs_size, risk_size).to(device)
		self.rb = ReplayBuffer()
		self.opt = optim.Adam(self.model.parameters(), lr=args.risk_lr)
		self.criterion = nn.NLLLoss()

	def train(self, num_epochs=10):
		self.model.train()
		data = self.rb.sample(self.args.num_risk_samples)
		dataset = RiskyDataset(data["next_obs"].to('cpu'), None, data["dist_to_fail"].to('cpu'), False, risk_type=self.args.risk_type,
								fear_clip=None, fear_radius=self.args.fear_radius, one_hot=True, quantile_size=self.args.quantile_size, quantile_num=self.args.quantile_num)
		dataloader = DataLoader(dataset, batch_size=self.args.risk_batch_size, shuffle=True, num_workers=4, generator=torch.Generator(device='cpu'))
		net_loss = 0
		for _ in tqdm.tqdm(range(num_epochs)):
			for batch in dataloader:
				pred = self.model(batch[0].to(self.device))
				loss = self.criterion(pred, torch.argmax(batch[1].squeeze(), axis=1).to(self.device))
				self.opt.zero_grad()
				loss.backward()
				self.opt.step()
				net_loss += loss.item()
		self.model.eval()
		return net_loss / (len(dataloader)*num_epochs)

	





def compute_fear(costs, max_dist=1000, type_="other"):
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
	if type_ == "fwd":
		return fear_fwd
	else:
		return torch.min(fear_fwd, fear_bwd)



class ReplayBuffer:
	def __init__(self, buffer_size=1e6):
		self.buffer_size = int(buffer_size)
		self.next_obs = None 
		self.risks = None 
		self.dist_to_fail = None 
		self.pos = 0
		self.full = False

	def add(self, next_obs, risk, dist_to_fail):
		# Initialize buffers if not already done
		if self.next_obs is None:
			# Determine the shape from the first observation
			obs_shape = next_obs.shape[1:] if len(next_obs.shape) > 1 else (next_obs.shape[0],)
			risk_shape = risk.shape[1:] if len(risk.shape) > 1 else (risk.shape[0],)
			dist_shape = dist_to_fail.shape[1:] if len(dist_to_fail.shape) > 1 else (dist_to_fail.shape[0],)
			
			self.next_obs = torch.zeros((self.buffer_size,) + obs_shape, dtype=next_obs.dtype, device=next_obs.device)
			self.risks = torch.zeros((self.buffer_size,) + risk_shape, dtype=risk.dtype, device=risk.device)
			self.dist_to_fail = torch.zeros((self.buffer_size,) + dist_shape, dtype=dist_to_fail.dtype, device=dist_to_fail.device)
		
		# Store data at current position
		self.next_obs[self.pos] = next_obs
		self.risks[self.pos] = risk
		self.dist_to_fail[self.pos] = dist_to_fail
		
		# Update position
		self.pos = (self.pos + 1) % self.buffer_size
		if self.pos == 0:
			self.full = True

	def __len__(self):
		if self.next_obs is None:
			return 0
		return self.buffer_size if self.full else self.pos

	def sample(self, sample_size):
		if self.next_obs is None or len(self) == 0:
			raise ValueError("Buffer is empty")
		
		# Sample indices
		max_idx = len(self)
		sample_idx = np.random.choice(max_idx, min(sample_size, max_idx), replace=False)
		
		return {
			"next_obs": self.next_obs[sample_idx],
			"risks": self.risks[sample_idx], 
			"dist_to_fail": self.dist_to_fail[sample_idx]
		}
	



class RiskyDataset(nn.Module):
    def __init__(self, obs, actions, risks, action=False, risk_type="discrete", fear_clip=None, fear_radius=None, one_hot=True, quantile_size=4, quantile_num=5):
        self.obs = obs
        self.risks = risks
        self.actions = actions
        self.one_hot = one_hot
        self.action = action
        self.fear_clip = fear_clip 
        self.fear_radius = fear_radius
        self.risk_type = risk_type

        self.quantile_size = quantile_size
        self.quantile_num = quantile_num

    def __len__(self):
        return self.obs.size()[0]
    
    def get_quantile_risk(self, idx):
        risk = self.risks[idx]
        y = torch.zeros(self.quantile_num)
        quant = self.quantile_size
        label = None
        for i in range(self.quantile_num-1):
            if risk < quant:
                label = i
                break
            else:
                quant += self.quantile_size
        if label is None:
            label = self.quantile_num-1

        y[label] = 1.0 
        return y

    def get_binary_risk(self, idx):
        if self.one_hot:
            y = torch.zeros(2)
            y[int(self.risks[idx] <= self.fear_radius)] = 1.0
        else:
            y = int(self.risks[idx] <= self.fear_radius)
        return y
    
    def get_continuous_risk(self, idx):
        if self.fear_clip is not None:
            return 1. / torch.clip(self.risks[idx]+1.0, 1, self.fear_clip)
        else:
            return 1. / self.risks[idx]

    def __getitem__(self, idx):
        if self.risk_type == "continuous":
            y = self.get_continuous_risk(idx)
        elif self.risk_type == "binary":
            y = self.get_binary_risk(idx)
        elif self.risk_type == "quantile":
            y = self.get_quantile_risk(idx)

        if self.action:
            return self.obs[idx], self.actions[idx], y
        else:
            return self.obs[idx], y