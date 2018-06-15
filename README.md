# LSTM sentiment analysis sandbox

## Things to do to run the demo
The project should use tensorflow-gpu, however tensorflow is sufficient fot running the demo.

To run `lstm_check_your_review_sentiment.py` you have to download [GoogleNews-vectors-negative300.bin](https://code.google.com/archive/p/word2vec/) [3,39 GB] and extract it to `data/google` subdirectory

## Running on Windows with python 3.5:
`pip install -r requirements.txt`

`set PYTHONPATH=.`

```
::imdb interactive review reviewer
python demo\lstm_check_your_review_sentiment.py
```
![reviewing your reviews](https://github.com/kasptom/ed/blob/master/assets/ed_review_demo.gif)
```
::imdb test
python demo\lstm_network_demo_binary.py
```
