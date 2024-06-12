import torch, sys, os, random, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import *
from model import *

n_hidden = 128
n_epochs = 100000
print_every = 100
plot_every = 1000
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    #category = randomChoice(all_categories)
    row = comp_fails.sample()
    line = row.iloc[0,row.columns.get_loc('name')] + " " + row.iloc[0,row.columns.get_loc('desc')]
    # expected_output = Variable(
    #     nn.functional.normalize(torch.tensor(
    #         [row.iloc[0,row.columns.get_loc('lower_bound')],
    #          row.iloc[0,row.columns.get_loc('best_estimate')],
    #          row.iloc[0,row.columns.get_loc('upper_bound')]]
    #         )))
    expected_output = Variable(
        torch.tensor(
            [row.iloc[0,row.columns.get_loc('lower_bound')],
             row.iloc[0,row.columns.get_loc('best_estimate')],
             row.iloc[0,row.columns.get_loc('upper_bound')]]
            ))
    #category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    #print(line)
    line_tensor = Variable(lineToTensor(row.iloc[0,row.columns.get_loc('name')] + " " + row.iloc[0,row.columns.get_loc('desc')]))
    return line, expected_output, line_tensor

rnn = RNN(n_letters, n_hidden, 3)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def train(expected_output, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output[0,:], expected_output)
    loss.backward()

    optimizer.step()

    return output, loss.data.item()

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    line, expected_output, line_tensor = randomTrainingPair()
    output, loss = train(expected_output, line_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        #guess, guess_i = categoryFromOutput(output)
        #correct = 'y' if guess == category else 'n (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, output, expected_output))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

