import torch, sys, os, random, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from data import *
import data
from model_lstm import *

class LSTMTrainer():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.NORMALIZATION_CONSTANT = 0.0001
        self.n_hidden = 256
        self.n_epochs = 100000
        self.print_every = 100
        self.plot_every = 1000
        self.learning_rate = 0.001 # If you set this too high, it might explode. If too low, it might not learn
        
        self.lstm = LSTM(data.n_letters,self.n_hidden,3).to(self.device)
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        return
    
    def randomChoice(self,l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingPair(self):
        #category = randomChoice(all_categories)
        row = data.comp_fails.sample()
        line = row.iloc[0,row.columns.get_loc('name')] + " " + row.iloc[0,row.columns.get_loc('desc')]
        # expected_output = Variable(
        #     nn.functional.normalize(torch.tensor(
        #         [row.iloc[0,row.columns.get_loc('lower_bound')],
        #          row.iloc[0,row.columns.get_loc('best_estimate')],
        #          row.iloc[0,row.columns.get_loc('upper_bound')]]
        #         )))
        expected_output = Variable(
            torch.mul(torch.tensor(
                [row.iloc[0,row.columns.get_loc('lower_bound')],
                row.iloc[0,row.columns.get_loc('best_estimate')],
                row.iloc[0,row.columns.get_loc('upper_bound')]]
                ,dtype=torch.float32).to(self.device),self.NORMALIZATION_CONSTANT))
        #category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
        #print(line)
        line_tensor = Variable(data.lineToTensor(row.iloc[0,row.columns.get_loc('name')] + " " + row.iloc[0,row.columns.get_loc('desc')],self.device))
        return line, expected_output, line_tensor

    def train(self,expected_output, line_tensor):
        self.optimizer.zero_grad()
        
        self.lstm.train()
        output= self.lstm(line_tensor)
        
        loss = self.criterion(output, expected_output)
        loss.backward()

        self.optimizer.step()
        return output, loss.data.item()

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def iterate(self):
        # Keep track of losses for plotting
        current_loss = 0
        all_losses = []
        
        start = time.time()
        
        for epoch in range(1, self.n_epochs + 1):
            line, expected_output, line_tensor = self.randomTrainingPair()
            output, loss = self.train(expected_output, line_tensor)
            current_loss += loss

            # Print epoch number, loss, name and guess
            if epoch % self.print_every == 0:
                #guess, guess_i = categoryFromOutput(output)
                #correct = 'y' if guess == category else 'n (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / self.n_epochs * 100, self.timeSince(start), loss, line, output, expected_output))
                print(current_loss/self.print_every)
                current_loss = 0

            # Add current loss avg to list of losses
            if epoch % self.plot_every == 0:
                all_losses.append(current_loss / self.plot_every)
                current_loss = 0

        torch.save(self.lstm, 'failure_curve_estimator.pt')
        

LSTMTrainer()