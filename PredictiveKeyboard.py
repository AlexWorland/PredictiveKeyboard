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
import random as random


# import matplotlib.pyplot as plt


def main():
    # Constants
    numSuggestions = 5
    # TODO: make user specifiable
    numPrevWords = 5
    epochs = 20
    optimizer = RMSprop(lr=0.01)
    createdText = ""

    uniqueWords, uniqueWordsIndex, prevWords, nextWords = prepareWords("", numPrevWords)

    # Would you like to load an existing model?
    if input("Would you like to load an existing model? (y/n): ") == 'y':
        modelPath = input("What is the file path for the model you wish to load? : ")
        historylPath = input("What is the file path for the model history you wish to load? : ")
        model, history = loadModel(modelPath, historylPath)
    else:
        epochs = int(input("How many epochs would you like to train for? : "))
        model, history, uniqueWords, uniqueWordsIndex = createNewModel(numPrevWords, optimizer, epochs)
        saveModelPrompt(model, history)

    print("Select a mode:")
    print("0 : Continuous")
    print("1 : N words")
    mode = int(input("Please enter a number:"))
    if mode == 1:
        length = int(input("Enter the length of words you would like to create: "))
        createdText = str(input("Please enter the first word: "))
        for i in range(length):
            suggestions = predictCompletions(createdText, model, uniqueWords, uniqueWordsIndex, numPrevWords,
                                             numSuggestions)

            selection = random.randrange(0,numSuggestions,1)
            createdText = createdText + " " + suggestions[selection]
        print(createdText)
        foo = 0
    createdText = str(input("Please enter the first word: "))
    choice = 0
    temp = ""
    while choice != -1:
        print(temp + createdText)
        suggestions = predictCompletions(createdText, model, uniqueWords, uniqueWordsIndex, numPrevWords,
                                         numSuggestions)
        print("Select one of the following (enter -1 to finish): ")
        for i in range(len(suggestions)):
            print(i, ":", suggestions[i])
        print(len(suggestions) + 1, ": End Sentence")
        print(len(suggestions) + 2, ": Enter custom word")
        choice = int(input("Please enter number: "))
        if choice == len(suggestions):
            createdText = createdText + "."
            temp = createdText
            createdText = ""
            createdText = " " + str(input("Please enter the start of the new sentence: "))
        elif choice == len(suggestions) + 1:
            createdText = createdText + str(input("Enter Word: "))
        elif choice > len(suggestions) + 1 or choice < -1:
            print("Please enter a number between ", -1, "and", len(suggestions)+2)
        else:
            createdText = createdText + " " + suggestions[choice]


def getDataset(path):
    """
    A function that returns the text from an input file
    :return: Returns the text from an input file
    """
    text = open(path, encoding='utf8').read().lower()
    print('corpus length: ', len(text))
    return splitDataset(text)


def saveModelPrompt(model, history):
    if input("Would you like to save the current model? (y/n): ") == 'y':
        fileName = input("What would you like to name the model file?")
        historyName = input("What would you like to name the history file?")
        saveModel(model, history, fileName, historyName)


def prepareWords(fileName, numPrevWords):
    # path = str(input("Please enter the file path of the input dataset: "))
    path = 'DANGER IN THE WHITE HOUSE.txt'
    data = getDataset(path)
    uniqueWords = np.unique(data)
    uniqueWordsIndex = dict((c, i) for i, c in enumerate(uniqueWords))
    previousWords = preparePrevWords(data, numPrevWords)
    nextWords = prepareNextWords(data, numPrevWords)
    return uniqueWords, uniqueWordsIndex, previousWords, nextWords


def createNewModel(numPrevWords, optimizer, epochs):
    # path = str(input("Please enter the file path of the input dataset: "))
    uniqueWords, uniqueWordsIndex, previousWords, nextWords = prepareWords("", numPrevWords)

    X, Y = prepareXYDatasets(previousWords, nextWords, numPrevWords, uniqueWords, uniqueWordsIndex)

    model = prepareModel(numPrevWords, uniqueWords)

    model = compileModel(model, optimizer)

    model, history = trainModel(model, X, Y, epochs)
    return model, history, uniqueWords, uniqueWordsIndex


def splitDataset(text):
    """
    A function that splits the input into individual words, stripped of special characters
    :param text: Input dataset
    :return: Returns a list of words
    """
    tokenizer = RegexpTokenizer(r'\w+[\.\,\?\;\:\'\"\!\?\(\)]?')
    words = tokenizer.tokenize(text)
    return words


def preparePrevWords(data, numPrevWords):
    previousWords = []
    for i in range(len(data) - numPrevWords):
        previousWords.append(data[i:i + numPrevWords])
    return previousWords


def prepareNextWords(data, numPrevWords):
    nextWords = []
    for i in range(len(data) - numPrevWords):
        nextWords.append(data[i + numPrevWords])
    return nextWords


def prepareXYDatasets(previousWords, nextWords, numPrevWords, uniqueWords, uniqueWordsIndex):
    X = np.zeros((len(previousWords), numPrevWords, len(uniqueWords)), dtype=bool)
    Y = np.zeros((len(nextWords), len(uniqueWords)), dtype=bool)
    for i, eachWords in enumerate(previousWords):
        for j, eachWord in enumerate(eachWords):
            X[i, j, uniqueWordsIndex[eachWord]] = 1
        Y[i, uniqueWordsIndex[nextWords[i]]] = 1
    return X, Y


def prepareModel(numPrevWords, uniqueWords):
    model = Sequential()
    model.add(LSTM(128, input_shape=(numPrevWords, len(uniqueWords))))
    model.add(Dense(len(uniqueWords)))
    model.add(Activation('softmax'))
    return model


def prepareInput(text, numPreviousWords, uniqueWords, uniqueWordsIndex):
    X = np.zeros((1, numPreviousWords, len(uniqueWords)))

    words = []
    for word in text.split():
        words.append(word)
    words.reverse()
    tmpText = ""
    for word in words:
        tmpText += word.lower() + " "

    counter = 4

    for t, word in enumerate(tmpText.split()):
        if counter < 0:
            break
        # print(word)
        if uniqueWordsIndex.__contains__(word):
            X[0, counter, uniqueWordsIndex[word]] = 1
        else:
            continue
        counter -= 1
    return X


def compileModel(model, optimizer):
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def trainModel(model, X, Y, epochs):
    history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=epochs, shuffle=True).history
    return model, history


def saveModel(model, history, fileName, historyName):
    model.save(fileName)
    pickle.dump(history, open(historyName, "wb"))


def loadModel(modelPath, historyPath):
    model = load_model(modelPath)
    history = pickle.load(open(historyPath, "rb"))
    return model, history


def sample(preds, top_n):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    expPreds = np.exp(preds)
    preds = expPreds / np.sum(expPreds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predictCompletions(text, model, uniqueWords, uniqueWordsIndex, numPreviousWords, n):
    if text == "":
        return 0
    x = prepareInput(text, numPreviousWords, uniqueWords, uniqueWordsIndex)
    preds = model.predict(x, verbose=0)[0]
    next_indicies = sample(preds, n)
    return [uniqueWords[idx] for idx in next_indicies]


if __name__ == '__main__':
    # import os
    #
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
    main()
