import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, seq_len, input_size, n_layers, hidden_size, n_labels=2, dropout=0.1, avg_pool=False):
        super(LSTMEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.avg_pool = avg_pool
        self.lstm_in = nn.LSTM(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout)   
        self.lstm_out = nn.LSTM(input_size=hidden_size, 
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout)   
        if avg_pool:
            self.avg_pool_layer =  nn.AvgPool1d(seq_len)
        self.linear = nn.Linear(hidden_size, n_labels)
        self.drop = nn.Dropout(p=dropout)
        
    def forward(self, x, device):
        h0 = torch.randn(self.n_layers, x.size()[0], self.hidden_size).to(device) 
        c0 = torch.randn(self.n_layers, x.size()[0], self.hidden_size).to(device)
        output, (hn, cn) = self.lstm_in(x, (h0, c0))      
        h0 = torch.randn(self.n_layers, x.size()[0], self.hidden_size).to(device)
        c0 = torch.randn(self.n_layers, x.size()[0], self.hidden_size).to(device)
        output, (hn, cn) = self.lstm_out(output, (h0, c0))
        if self.avg_pool:
            output = self.avg_pool_layer(output)
        else:
            output = output[:,-1,:]
        output = self.drop(self.linear(output))
        return output

