
# Time Difference Embeddings
Predicting the temporal difference between two dates in hours by the LSTM seq2seq model.

## Installation
Clone this repository:

```
git clone https://github.com/hajipoor/time2event.git
cd time2event/diff_model
```
Install required packages:
```bash
pip install -r requirements.txt
```

## Train
To train a model:
```bash
python train_diff_model.py
```
## Test
To use the pre-trained model:
```bash
python test_diff_model.py data/dates.txt data/embeddings.txt
```
_dates.txt_ is comma separated file of desired dates and their time difference embedding vectors will be saved into the _embeddings.txt_ file. Please see data/dates.txt and data/embeddings.txt

>This figure shows **Time  differences**  in  the  embedding space such that each sample point shows the embedding of time difference between random t<sub>1</sub> and t<sub>2</sub>. If _t<sub>1</sub> − t<sub>2</sub> = t′<sub>1</sub> − t′<sub>2</sub>_, their diff embeddings are close. See our [paper](https://2021.aclweb.org/) for details on the model.

![Time Difference Embeddings](https://github.com/hajipoor/time2event/raw/main/diff_model/data/embeddings.png)


