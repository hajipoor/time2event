import argparse
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

np.random.seed(2021)

START_YEAR = 2015
END_YEAR = 2021
EPOCHS = 150
BATCH_SIZE = 32
MIN_NUM_SAMPLES = (END_YEAR - START_YEAR) * 365 * 24
OUTPUT = 'models'

# All the numbers, minus sign and space for padding.
CHARS = "0123456789- "

# Maximum number of digits to present date in hours for example '2020/09/01' presents with 6 digits '444144' hours
DIGITS = 6

# Maximum length of input is 'date1 - date2' in hours format (e.g., '444144 - 155328').
MAXLEN = 2 * DIGITS + 1


class CharacterTable:
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.
        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        print(C)
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.
        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)


def generate_data():
    print("Generating data...")

    num_samples_per_delta = 10
    seen = set()
    samples, labels = [], []

    for _ in range(MIN_NUM_SAMPLES):
        delta = timedelta(hours=int(np.random.choice(list(range(0, MIN_NUM_SAMPLES)))))

        for i in range(num_samples_per_delta):
            def generate_random_date():
                while True:
                    try:
                        return datetime(np.random.choice(list(range(START_YEAR, END_YEAR + 1))),
                                        np.random.choice(list(range(1, 13))),
                                        np.random.choice(list(range(1, 31))), np.random.choice(list(range(1, 24))))
                    except:
                        pass

            a = generate_random_date()
            b = a - delta
            if b < datetime(START_YEAR, 1, 1, 1):
                b = a + delta
                a, b = b, a

            b = b.replace(tzinfo=timezone.utc)
            b = int(b.timestamp() / 3600)

            a = a.replace(tzinfo=timezone.utc)
            a = int(a.timestamp() / 3600)

            diff_hours = a - b
            key = '{}_{}'.format(a, b)

            if key in seen:
                i = i - 1
                continue
            seen.add(key)

            # Pad the input with spaces such that it is always MAXLEN.
            q = "{}-{}".format(a, b)
            query = q + " " * (MAXLEN - len(q))

            label = str(diff_hours)
            # label can be maximum size of DIGITS.
            label += " " * (DIGITS - len(label))

            samples.append(query)
            labels.append(label)

    print("Total samples:", len(samples))

    return samples, labels


def create_model():
    print("Build model...")

    model = keras.Sequential()
    # "Encode" the input sequence using a LSTM, producing an output of size 128.
    model.add(layers.LSTM(128, input_shape=(MAXLEN, len(CHARS))))

    # As the decoder RNN's input, repeatedly provide with the last output of
    # RNN for each time step. Repeat 'DIGITS' times as that's the maximum length of output
    model.add(layers.RepeatVector(DIGITS))

    # The decoder RNN layer.
    model.add(layers.LSTM(64, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step
    # of the output sequence, decide which character should be chosen.
    model.add(layers.Dense(len(CHARS), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


def train():
    char_table = CharacterTable(CHARS)
    samples, labels = generate_data()

    print("Vectorization...")
    x = np.zeros((len(samples), MAXLEN, len(CHARS)), dtype=np.bool)
    y = np.zeros((len(samples), DIGITS, len(CHARS)), dtype=np.bool)
    for i, sample in enumerate(samples):
        x[i] = char_table.encode(sample, MAXLEN)
    for i, label in enumerate(labels):
        y[i] = char_table.encode(label, DIGITS)

    # Explicitly set apart 10% for validation data that we never train over.
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    model = create_model()

    # Create output folder
    Path(os.path.join(OUTPUT, "diff_model")).mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(os.path.join(OUTPUT, "diff_model"), monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    # Train the model .
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val), callbacks=[checkpoint, early_stop]
    )


if __name__ == "__main__":
    train()
    print('Model saved into {}'.format(os.path.join(OUTPUT, "diff_model")))