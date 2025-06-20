# Implementation of Masked Auto-regressive Flow (MAF) model trained on moons dataset.
# MAF is implemented using 5 flows, each flow modelled as Masked Auto-regressive Density Estimator (MADE)
# with one hidden layer.

import csv
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MaskedLinear(nn.Linear):
    """Implements a linear layer with masking."""

    def __init__(self, input_size, output_size, mask):
        super().__init__(input_size, output_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)
    

class PermuteLayer(nn.Module):
    """Flips ordering of input and returns it.
    An extra output of all zeros is returned so that number of outputs between
    MADE and PermuteLayer are same, so that they can be chained sequentially.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.flip(input, -1)
    
    def inverse(self, input):
        return torch.flip(input, [-1]), torch.zeros(input.shape[0], 1)
    
    

class MADE(nn.Module):
    """Implements single layer of MADE."""

    def __init__(self, n_hidden, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        masks = self.create_masks(n_hidden, input_size, hidden_size)
        layers = [MaskedLinear(input_size, hidden_size, masks[0])]

        for idx in range(n_hidden - 1):
            l = MaskedLinear(hidden_size, hidden_size, masks[idx + 1])
            layers.append(l)

        l = MaskedLinear(hidden_size, 2*input_size, masks[-1])
        layers.append(l)

        self.net = nn.Sequential(*layers)


    def create_masks(self, n_hidden, input_size, hidden_size):
        # input activation
        # indices refers to index assigned to node
        indices = [torch.arange(0, input_size)]

        # hidden activation
        for h in range(n_hidden):
            # (input_size -1) since we don't want input to be copied to output
            t = (h + torch.arange(0, hidden_size)) % (input_size - 1)    
            indices.append(t)

        # output activation
        t = torch.arange(0, 2*input_size)
        indices.append(t)

        # for temp in range(len(indices)):
        #     print(f'indices[{temp}]: {indices[temp]}')

        # generate masks 
        # if index of output <= index of input then mask = 1, otherwise mask = 0
        d_in = indices[:-1]
        d_out = indices[1:]
        masks = []

        for i in range(n_hidden + 1):
            m = (d_out[i].unsqueeze(-1) >= d_in[i].unsqueeze(0)).float()
            masks.append(m)

        return masks
    

    def inverse(self, x):
        """Defines the inverse function for MADE : z = (x-mu)* exp(-sigma)"""

        t = self.net(x)
        mu = t[:, :self.input_size]
        sigma = t[:, self.input_size:]
        z = (x - mu) * torch.exp(-sigma)
        log_det = -torch.sum(sigma, dim=-1)
        return z, log_det
    
    
    def forward(self, z):
        """Defines the forward function for MADE: x = mu + z*exp(sigma)"""

        x = torch.zeros_like(z)

        for i in range(self.input_size):
            t = self.net(x)
            mu = t[:, :self.input_size]
            sigma = t[:, self.input_size:]
            x = mu + z * torch.exp(sigma)

        return x
    

class MAF(nn.Module):
    """Implementation of Masked Auto-regressive Flow (MAF) using multiple flow networks."""

    def __init__(self, num_flows, input_size, hidden_size, n_hidden):
        super().__init__()
        self.flows = []
        self.input_size = input_size

        for _ in range(num_flows - 1):
            self.flows.append(MADE(n_hidden, input_size, hidden_size))
            self.flows.append(PermuteLayer())

        self.flows.append(MADE(n_hidden, input_size, hidden_size))
        self.flow_model = nn.Sequential(*self.flows)
        self.base_dist = torch.distributions.normal.Normal(0, 1)

    
    def log_prob(self, x):
        batch_size = x.shape[0]
        sum_log_det = torch.zeros(batch_size, dtype=torch.float)

        for flow in self.flow_model:
            z, log_det = flow.inverse(x)
            sum_log_det += torch.squeeze(log_det)
            x = z

        first_term = torch.sum(self.base_dist.log_prob(x), dim=-1)
        final_logprob = torch.sum(first_term + sum_log_det) / batch_size
        return final_logprob
    
    
    def loss(self, x):
        return -self.log_prob(x)
    
    
    def sample(self, batch_size=1):
        z = torch.randn(batch_size, self.input_size)

        for flow in self.flow_model:
            z = flow.forward(z)

        return z.numpy()


        
class MoonsDataset(Dataset):
    """Moons dataset from Kaggle (created using make_moons() api from Sklearn).
    Contains the (x,y) positions of two non-intersecting semi-circles.
    """

    def __init__(self, path, is_train: bool):
        super().__init__()
        self.path = path
        self.train_split = 0.8
        self.is_train = is_train
        
        with open(path, 'r', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            self.rows = list(reader)

    
    def __len__(self):
        if self.is_train:
            return int(self.train_split * len(self.rows))
        else:
            return int((1 - self.train_split) * len(self.rows))
        
    
    def __getitem__(self, index):
        if not self.is_train:
            index = int(self.train_split * len(self.rows)) + 1 + index
            
        row = self.rows[index]
        return torch.tensor([float(row['X1']), float(row['X2'])], dtype=torch.float)
    

def train(maf: MAF, path: str, batch_size: int, num_epochs: int):
    maf.train()
    train_dataset = MoonsDataset(path, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(maf.parameters(), lr=1e-3)

    val_dataset = MoonsDataset(path, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        # train
        for batch, x in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = maf.loss(x)
            print(f'epoch: {epoch} batch: {batch}, train_loss: {loss}')
            
            loss.backward()
            optimizer.step()

        # validation
        maf.eval()
        loss = 0.0
        for batch, x in enumerate(val_dataloader):
            loss += maf.loss(x)

        loss = loss / len(val_dataloader)
        print(f'val_loss: {loss}')
        
        

def main():
    # model params
    input_size = 2
    hidden_size = 10
    n_hidden = 1
    num_flows = 5

    # training params
    data_path = '/kaggle/input/sklearn-moons-data-set/cluster_moons.csv'
    batch_size = 4
    num_epochs = 2

    maf = MAF(num_flows, input_size, hidden_size, n_hidden)
    train(maf, path, batch_size, num_epochs)


    

if __name__ == "__main__":
    main()

