import torch, glob, unicodedata, string
import pandas as pd

all_letters = string.ascii_letters + " .,;'-"
#print(all_letters)
n_letters = len(all_letters)

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# TODO: change source
comp_fails = pd.read_csv(filepath_or_buffer="lstm/tmp_db")

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line,device):
    tensor = torch.zeros(len(line), 1, n_letters,dtype=torch.float32).to(device)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

