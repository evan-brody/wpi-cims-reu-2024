import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size,hidden_size,self.num_layers)
        self.act = nn.Linear(hidden_size,output_size)

    def forward(self, input):
        output,_=self.lstm(input)
        output = self.act(output)
        return output

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
