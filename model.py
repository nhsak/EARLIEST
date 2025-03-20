import torch
from torch import nn
from modules import Controller, BaselineNetwork
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super(CNNFeatureExtractor, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to get fixed size output
        )
        
    def forward(self, x):
        # Input shape: [C, B, T, H, W] = [3, 8, 80, 190, 20]
        c, b, t, h, w = x.shape
        
        # Reshape and process each time step through CNN
        features = []
        for i in range(t):
            # Get current time step: [B, C, H, W]
            curr_x = x[:, :, i, :, :].permute(1, 0, 2, 3)
            
            # Pass through CNN: [B, 64, 1, 1]
            curr_features = self.cnn(curr_x)
            
            # Reshape to [B, 64]
            curr_features = curr_features.view(b, -1)
            
            # Collect features
            features.append(curr_features)
        
        # Stack features: [T, B, 64]
        features = torch.stack(features)

        return features




class EARLIEST(nn.Module):
    def __init__(self, ninp, nclasses, args):
        super(EARLIEST, self).__init__()

        # Hyperparameters
        self.nclasses = nclasses
        self.rnn_cell = args.rnn_cell
        self.nhid = args.nhid
        self.nlayers = args.nlayers
        self.lam = args.lam

        # Update intransforms in EARLIEST model
        self.intransforms = CNNFeatureExtractor(3, self.nhid)

        # Sub-networks
        self.Controller = Controller(self.nhid+1, 1)
        self.BaselineNetwork = BaselineNetwork(self.nhid+1, 1)

        if self.rnn_cell == "LSTM":
            self.RNN = torch.nn.LSTM(self.nhid, self.nhid, num_layers=self.nlayers)
        elif self.rnn_cell == "GRU":
            self.RNN = torch.nn.GRU(self.nhid, self.nhid)
        else:
            self.RNN = torch.nn.RNN(self.nhid, self.nhid)

        self.out = torch.nn.Linear(self.nhid, self.nclasses)

        # Move the model to the correct device
        self.to(device)

    def initHidden(self, bsz):
        if self.rnn_cell == "LSTM":
            return (torch.zeros(self.nlayers, bsz, self.nhid).to(device),
                    torch.zeros(self.nlayers, bsz, self.nhid).to(device))
        else:
            return torch.zeros(self.nlayers, bsz, self.nhid).to(device)

    def forward(self, X, epoch=0, test=False):
        if test:
            self.Controller._epsilon =  1
        else:
            self.Controller._epsilon = 1 # set explore/exploit trade-off

        X = X.to(device)  # Ensure input is on the correct device
        X = self.intransforms(X)
        print(X.shape)
        T, B, V = X.shape
        baselines = []
        actions = []
        log_pi = []
        halt_probs = []
        halt_points = -torch.ones((B, self.nclasses)).to(device)
        hidden = self.initHidden(X.shape[1])
        predictions = torch.zeros((B, self.nclasses), requires_grad=True).to(device)
        all_preds = []

        for t in range(T):
            x = X[t].unsqueeze(0)
            output, hidden = self.RNN(x, hidden)
            logits = self.out(output.squeeze())

            time = torch.tensor([t], dtype=torch.float, requires_grad=False).view(1, 1, 1).repeat(1, B, 1).to(device)
            c_in = torch.cat((output, time), dim=2).detach()
            a_t, p_t, w_t = self.Controller(c_in)

            predictions = torch.where((a_t == 1) & (predictions == 0), logits, predictions)
            halt_points = torch.where((halt_points == -1) & (a_t == 1), time.squeeze(0), halt_points)
            b_t = self.BaselineNetwork(torch.cat((output, time), dim=2).detach())

            actions.append(a_t.squeeze())
            baselines.append(b_t.squeeze())
            log_pi.append(p_t)
            halt_probs.append(w_t)

            if (halt_points == -1).sum() == 0:
                break

        logits = torch.where(predictions == 0.0, logits, predictions).squeeze()
        halt_points = torch.where(halt_points == -1, time, halt_points).squeeze(0)
        self.locations = np.array(halt_points.cpu() + 1)
        self.baselines = torch.stack(baselines).squeeze(1).transpose(0, 1)
        self.log_pi = torch.stack(log_pi).squeeze(1).squeeze(2).transpose(0, 1)
        self.halt_probs = torch.stack(halt_probs).transpose(0, 1).squeeze(2)
        self.actions = torch.stack(actions).transpose(0, 1)

        self.grad_mask = torch.zeros_like(self.actions)
        for b in range(B):
            self.grad_mask[b, :(1 + halt_points[b, 0]).long()] = 1
        return logits.squeeze(), (1 + halt_points).mean() / (T + 1)

    def computeLoss(self, logits, y):
        _, y_hat = torch.max(torch.softmax(logits, dim=1), dim=1)
        self.r = (2 * (y_hat.float().round() == y.float()).float() - 1).detach().unsqueeze(1).to(device)
        self.R = self.r * self.grad_mask.to(device)

        b = self.grad_mask * self.baselines.to(device)
        self.adjusted_reward = self.R - b.detach()

        MSE = torch.nn.MSELoss()
        CE = torch.nn.CrossEntropyLoss()
        self.loss_b = MSE(b, self.R)
        self.loss_r = (-self.log_pi * self.adjusted_reward).sum() / self.log_pi.shape[1]
        self.loss_c = CE(logits, y)
        self.wait_penalty = self.halt_probs.sum(1).mean()
        self.lam = torch.tensor([self.lam], dtype=torch.float, requires_grad=False).to(device)
        loss = self.loss_r + self.loss_b + 3 * self.loss_c + self.lam * (self.wait_penalty)
        return loss
