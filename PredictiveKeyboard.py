"""
Author: Alex Worland
Date: 10/31/2020
File: PredictiveKeyboard.py
Description:
    A predictive keyboard that can be built with a user input dataset
"""

# Dependencies
import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import pickle
import heapq
import matplotlib.pyplot as plt


def main():
    data = getDataset()
    uniqueWords = np.unique(data)
    uniqueWordsIndex = dict((c, i) for i, c in enumerate(uniqueWords))

    # TODO: make user specifiable
    numPrevWords = 5
    previousWords = []
    nextWords = []
    for i in range(len(data - numPrevWords)):
        previousWords.append(data[i:i + numPrevWords])
        nextWords.append(data[i + numPrevWords])
    print(previousWords[0])
    print(nextWords[0])

    X = np.zeros((len(previousWords), numPrevWords, len(uniqueWords)), dtype=bool)
    Y = np.zeros((len(nextWords), len(uniqueWords)), dtype=bool)
    for i, eachWords in enumerate(previousWords):
        for j, eachWord in enumerate(eachWords):
            X[i:j, uniqueWordsIndex[eachWord]] = 1
        Y[i, uniqueWordsIndex[nextWords[i]]] = 1
    print(X[0][0])

    model = Sequential()
    model.add(LSTM(128, input_shape=(numPrevWords, len(previousWords))))
    model.add(Dense(len(uniqueWords)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

    model.save('kerasNextWordModel.h5')
    pickle.dump(history, open("history.p", "wb"))
    model = load_model('kerasNextWordModel.h5')
    history = pickle.load(open("history.p", "rb"))

    q = "Your life will never be the same again."
    print("Correct Sentence: ", q)
    seq = " ".join(RegexpTokenizer(r'\w+').tokenize(q.lower())[0:numPrevWords])
    print("Sequence: ", seq)
    print("next possible words: ", predictCompletions(seq, model, uniqueWords))


def getDataset():
    """
    A function that returns the text from an input file
    :return: Returns the text from an input file
    """
    path = input("Please enter the file path of the input dataset: ")
    text = open(path).read().lower()
    print('corpus length: ', len(text))
    return splitDataset(text)


def splitDataset(text):
    """
    A function that splits the input into individual words, stripped of special characters
    :param text: Input dataset
    :return: Returns a list of words
    """
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    return words


def prepareInput(text, numPreviousWords, uniqueWords, uniqueWordsIndex):
    X = np.zeros((1, numPreviousWords, len(uniqueWords)))
    for t, word in enumerate(text.split()):
        print(word)
        X[0, t, uniqueWordsIndex[word]] = 1
    return X


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    expPreds = np.exp(preds)
    preds = expPreds / np.sum(expPreds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predictCompletions(text, model, uniqueWords, n=3):
    if text == "":
        return 0
    x = prepareInput(text)
    preds = model.predict(x, verbose=0)[0]
    next_indicies = sample(preds, n)
    return [uniqueWords[idx] for idx in next_indicies]
