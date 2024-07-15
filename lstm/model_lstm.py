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
        output_packed,_=self.lstm(input)
        output, lengths = turnn.pad_packed_sequence(output_packed, batch_first=True)
        
        output = self.act(self.extract_packed_sequence(output,lengths))
        return torch.flatten(output)

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
    
    def extract_packed_sequence(self,output, lengths):
        e = torch.empty(0)
        for i in range(len(lengths)):
            e = torch.cat((e,output[i][lengths[i]-1][:].reshape(1,self.hidden_size)))
        return e
