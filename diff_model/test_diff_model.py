from tensorflow.keras import Model
import keras
import datetime
import numpy as np
import argparse
import os
from pathlib import Path

from train_diff_model import OUTPUT, MAXLEN, CHARS, CharacterTable

parser = argparse.ArgumentParser(description='diff model')

parser.add_argument('input', type=str, default='data/dates.txt', help='input file in comma separate format date1,date2')
parser.add_argument('output', type=str, default='data/embeddings.txt', help='output file for saving embeddings')

args = parser.parse_args()


def load_model():
    model = keras.models.load_model(os.path.join(OUTPUT, "diff_model"))
    inp = model.input
    out = model.layers[-2].output
    return Model(inp, out)


def read_dates(path):
    total_dates = []
    for line in open(path):

        date1, date2 = line.split(',')
        a = datetime.datetime.strptime(date1.strip(), '%Y-%m-%d')
        b = datetime.datetime.strptime(date2.strip(), '%Y-%m-%d')

        if a < b:
            a, b = b, a

        b = b.replace(tzinfo=datetime.timezone.utc)
        b = int(b.timestamp() / 3600)

        a = a.replace(tzinfo=datetime.timezone.utc)
        a = int(a.timestamp() / 3600)

        # Pad the input with spaces such that it is always MAXLEN.
        q = "{}-{}".format(a, b)
        total_dates.append(q + " " * (MAXLEN - len(q)))

    # Vectorization
    char_table = CharacterTable(CHARS)
    x = np.zeros((len(total_dates), MAXLEN, len(CHARS)), dtype=np.bool)
    for i, sample in enumerate(total_dates):
        x[i] = char_table.encode(sample, MAXLEN)

    return x


# Load pre-trained model
diff_model = load_model()

# Read inputs from file
samples = read_dates(args.input)

# Predict
pred_vectors = diff_model.predict(samples, verbose=1)

Path(args.output).mkdir(parents=True, exist_ok=True)

# Save embeddings to file
with open(args.output, 'w') as f:
    for pred_vec in pred_vectors:
        f.write(' '.join(map(str, pred_vec.flatten())))
        f.write('\n')

print('Embeddings vectors saved into {}'.format(args.output))
