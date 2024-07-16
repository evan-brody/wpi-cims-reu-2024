import torch, sys, os, random, time, math, unicodedata, string, copy, gui.gui as gui
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from lstm.model_lstm import *
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as turnn

ALL_LETTERS = string.ascii_letters + " .,;'-"
N_LETTERS = len(ALL_LETTERS)

NORMALIZATION_CONSTANT = 0.01
N_HIDDEN = 512
N_EPOCHS = 1000
EPOCH_SIZE = 10
LEARNING_RATE = 0.0001 # If you set this too high, it might explode. If too low, it might not learn

PLOT_UPDATE_INTERVAL = 4

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

if __name__ == "__main__":
    window = gui.window
    pool = gui.pool
    lstm = LSTM(N_LETTERS,N_HIDDEN,1).to(device)
    lstm.share_memory()
    best_model = LSTM(N_LETTERS,N_HIDDEN,1).to(device)
    lowest_error = 3.402823466E+38

if __name__ == "lstm.train_lstm":
    lstm = LSTM(N_LETTERS,N_HIDDEN,1).to(device)
    lstm.share_memory()
    best_model = LSTM(N_LETTERS,N_HIDDEN,1).to(device)
    lowest_error = 3.402823466E+38
######## DATA LOADING ########

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    row = comp_fails.sample()
    #line = row.iloc[0,row.columns.get_loc('name')] + "," + row.iloc[0,row.columns.get_loc('desc')]
    line = row.iloc[0,row.columns.get_loc('name')]
    expected_output = Variable(
        torch.mul(torch.tensor(
            [row.iloc[0,row.columns.get_loc('best_estimate')]],dtype=torch.float32),NORMALIZATION_CONSTANT)).to(device)
    #line_tensor = Variable(lineToTensor(row.iloc[0,row.columns.get_loc('name')] + "," + row.iloc[0,row.columns.get_loc('desc')]))
    line_tensor = Variable(lineToTensor(row.iloc[0,row.columns.get_loc('name')]))
    return line, expected_output, line_tensor

def random_training_pair_batched():
    row = comp_fails.sample()
    #line = row.iloc[0,row.columns.get_loc('name')] + "," + row.iloc[0,row.columns.get_loc('desc')]
    line = row.iloc[0,row.columns.get_loc('name')]
    expected_output = Variable(
        torch.mul(torch.tensor(
            [row.iloc[0,row.columns.get_loc('best_estimate')]],dtype=torch.float32),NORMALIZATION_CONSTANT)).to(device)
    #line_tensor = Variable(lineToTensor(row.iloc[0,row.columns.get_loc('name')] + "," + row.iloc[0,row.columns.get_loc('desc')]))
    line_tensor = Variable(line_to_tensor_2d(row.iloc[0,row.columns.get_loc('name')]))
    return line, expected_output, line_tensor

def gen_batched_training_pairs():
    lines = []
    e_outs = torch.empty(0)
    l_ts_lst = []
    max_size = 0
    
    for i in range(EPOCH_SIZE):
        line, expected_output, line_tensor = random_training_pair_batched()
        
        if(line_tensor.size()[0]>max_size):
            max_size = line_tensor.size()[0]
        
        lines.append(line)
        e_outs = torch.cat((e_outs,expected_output))
        l_ts_lst.append(line_tensor)
    
    padded = turnn.pad_sequence(l_ts_lst, batch_first=True, padding_value=0.0)

    lengths = torch.tensor([len(t) for t in l_ts_lst])
    packed = turnn.pack_padded_sequence(padded, lengths.to(device), batch_first=True, enforce_sorted=False)

    return lines,e_outs,packed

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
def lineToTensor(line: str):
    tensor = torch.zeros(1,len(line), N_LETTERS,dtype=torch.float32).to(device)
    for li, letter in enumerate(line):
        tensor[0][li][letterToIndex(letter)] = 1
    return tensor

def line_to_tensor_2d(line: str):
    tensor = torch.zeros(len(line), N_LETTERS,dtype=torch.float32).to(device)
    for li, letter in enumerate(line):
        tensor[li][letterToIndex(letter)] = 1
    return tensor

######## END OF DATA LOADING ########

def save_model():
    torch.save(best_model, os.path.join(os.path.dirname(__file__),'failure_rate_estimator.pt'))

def load_model():
    global best_model
    best_model = torch.load(os.path.join(os.path.dirname(__file__),'failure_rate_estimator.pt'))
    window.update_prediction()

def train_batched(args):
    model,e_outs,packed_l_ts = args
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    model.train()
    output= model.forward_batched(packed_l_ts)
    
    loss = criterion(output, e_outs)
    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    return output, loss.data.item()
    

def train(expected_output, line_tensor):
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    lstm.train()
    output= lstm.forward(line_tensor)
    
    loss = criterion(output, expected_output)
    optimizer.zero_grad()
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
    #return [avg_loss,time.time(),epoch, line, output, expected_output]

def load_batch(args):
    model,epoch = args
    lines,e_outs,packed_l_ts = gen_batched_training_pairs()
    outs, avg_loss = train_batched((model,e_outs,packed_l_ts))
    
    return [avg_loss,time.time(),epoch]

def start_training():
    # Keep track of losses for plotting
    r = random.random() 
    b = random.random() 
    g = random.random() 
    
    window.loss_fig_color = (r, g, b) 
    window.start_time = time.time()
    window.loss_x = [0]
    window.loss_y = [1]
    # plotting the first frame
    window.loss_fig = plt.plot(window.loss_x,window.loss_y)[0]
    plt.ylim(0,1)
    
    for i in range(N_EPOCHS):
        #pool.apply_async(iterate,args=[i],callback=async_callback)
        pool.apply_async(load_batch,args=[(lstm,i)],callback=async_callback)
    #pool.apply_async(train,args=[0,trainer],callback=async_callback)

def stop_training():
    pool.terminate()
    #pool.close()
    return

def async_callback(func_result):
    print(func_result)
    global lowest_error
    # updating the data
    #line,output,expected_output,loss,avg_loss,del_time,epoch = func_result
    window.loss_x.append(func_result[1]-window.start_time)
    window.loss_y.append(func_result[0])
    #print('{d} {d}% ({s}) {.4f} {s} / {s} {s}'.format(epoch, epoch / 10, x[-1], loss, line, output, expected_output))
    if(func_result[0]<lowest_error):
        lowest_error = func_result[0]
        window.min_loss_box.setText(str(lowest_error))
        global best_model
        best_model = copy.deepcopy(lstm)
    # removing the older graph
    if(func_result[2] % PLOT_UPDATE_INTERVAL ==0):
        window.update_prediction()
        window.loss_fig.remove()
        
        # # plotting newer graph
        with plt.ion():
            window.loss_fig = plt.plot(window.loss_x,window.loss_y,color= window.loss_fig_color)[0]
            
            plt.xlim(window.loss_x[0], window.loss_x[-1])
            plt.ylim(0, max(window.loss_y))


def predict(line: str) -> torch.Tensor:
    lt = [line_to_tensor_2d(line)]
    
    padded = turnn.pad_sequence(lt, batch_first=True, padding_value=0.0)

    lengths = torch.tensor([len(t) for t in lt])
    packed = turnn.pack_padded_sequence(padded, lengths.to(device), batch_first=True, enforce_sorted=False)
    best_model.eval()
    return best_model.forward_batched(packed)
    # line_tensor = lineToTensor(line)
    # return lstm.forward(line_tensor)