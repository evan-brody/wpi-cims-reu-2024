import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.rnn as turnn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size,hidden_size,self.num_layers,batch_first=True)
        self.act = nn.Linear(hidden_size,output_size)

    def forward(self, input):
        output,_=self.lstm(input)
        
        output = self.act(output[:,-1,:])
        return torch.flatten(output)
    
    def forward_batched(self, input):
        output,_=self.lstm(input)
        output, lengths = turnn.pad_packed_sequence(output, batch_first=True)
        
        output = self.act(output[:,-1,:])
        return torch.flatten(output)

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
