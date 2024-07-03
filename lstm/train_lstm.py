import torch, sys, os, random, time, math, unicodedata, string
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from lstm.model_lstm import *
import matplotlib.pyplot as plt

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

ALL_LETTERS = string.ascii_letters + " .,;'-"
N_LETTERS = len(ALL_LETTERS)

NORMALIZATION_CONSTANT = 0.0001
N_HIDDEN = 256
N_EPOCHS = 100
EPOCH_SIZE = 10
LEARNING_RATE = 0.001 # If you set this too high, it might explode. If too low, it might not learn

lstm = LSTM(N_LETTERS,N_HIDDEN,3).to(device)
lstm.share_memory()
optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

######## DATA LOADING ########

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    #category = randomChoice(all_categories)
    row = comp_fails.sample()
    line = row.iloc[0,row.columns.get_loc('name')] + " " + row.iloc[0,row.columns.get_loc('desc')]
    expected_output = Variable(
        torch.mul(torch.tensor(
            [row.iloc[0,row.columns.get_loc('lower_bound')],
            row.iloc[0,row.columns.get_loc('best_estimate')],
            row.iloc[0,row.columns.get_loc('upper_bound')]]
            ,dtype=torch.float32),NORMALIZATION_CONSTANT)).to(device)
    line_tensor = Variable(lineToTensor(row.iloc[0,row.columns.get_loc('name')] + " " + row.iloc[0,row.columns.get_loc('desc')],device))
    return line, expected_output, line_tensor

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# TODO: change source
comp_fails = pd.read_csv(filepath_or_buffer="lstm/tmp_db")

# Find letter index from ALL_LETTERS, e.g. "a" = 0
def letterToIndex(letter):
    return ALL_LETTERS.find(letter)

# Turn a line into a <line_length x 1 x N_LETTERS>,
# or an array of one-hot letter vectors
def lineToTensor(line,device):
    tensor = torch.zeros(len(line), 1, N_LETTERS,dtype=torch.float32).to(device)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

######## END OF DATA LOADING ########

def train(expected_output, line_tensor):
    optimizer.zero_grad()
    
    lstm.train()
    output= lstm(line_tensor)
    
    loss = criterion(output, expected_output)
    loss.backward()

    optimizer.step()
    return output, loss.data.item()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def iterate_once():
    line, expected_output, line_tensor = randomTrainingPair()
    output, loss = train(expected_output, line_tensor)
    return line,output,expected_output,loss

def iterate(epoch):
    current_loss = 0
    for e in range(EPOCH_SIZE):
        line,output,expected_output,loss = iterate_once()
        current_loss += loss
    
    avg_loss = current_loss/EPOCH_SIZE
    # Print epoch number, loss, name and guess
    return [avg_loss,time.time(),epoch]

def start_training():
    # Keep track of losses for plotting
    
    window.start_time = time.time()
    window.loss_x = [0]
    window.loss_y = [1]
    # plotting the first frame
    window.loss_fig = plt.plot(window.loss_x,window.loss_y)[0]
    plt.ylim(0,1)
    
    for i in range(N_EPOCHS):
        pool.apply_async(iterate,args=[i],callback=async_callback)
    #pool.apply_async(train,args=[0,trainer],callback=async_callback)

def stop_training():
    pool.terminate()
    #pool.close()
    return

def async_callback(func_result):
    print(func_result)
    # updating the data
    #line,output,expected_output,loss,avg_loss,del_time,epoch = func_result
    window.loss_x.append(func_result[1]-window.start_time)
    window.loss_y.append(func_result[0])
    #print('{d} {d}% ({s}) {.4f} {s} / {s} {s}'.format(epoch, epoch / 10, x[-1], loss, line, output, expected_output))
    # removing the older graph
    window.loss_fig.remove()
    
    # # plotting newer graph
    with plt.ion():
        window.loss_fig = plt.plot(window.loss_x,window.loss_y,color = 'g')[0]
        plt.xlim(window.loss_x[0], window.loss_x[-1])
        plt.ylim(0, max(window.loss_y))
