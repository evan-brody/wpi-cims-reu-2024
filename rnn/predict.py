from model import *
from data import *
import sys

rnn = torch.load('failure_curve_estimator.pt')

# Just return an output given a line
def evaluate(line_tensor):
    for i in range(line_tensor.size()[0]):
        output= rnn(line_tensor[i])
    
    return output

def predict(line, n_predictions=3):
    output = evaluate(Variable(lineToTensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

if __name__ == '__main__':
    predict(sys.argv[1])
