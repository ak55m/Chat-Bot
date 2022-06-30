from urllib import response
import nltk 
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn 
import tensorflow
import random
import json
import pickle

#opening the json file
with open("intents.json") as file:
    data = json.load(file)

#for getting only necessary data lets say when we dont have to run the code to train the AI model then do that

try: #try to open up this file and if it can't then it does the except
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)    

except:
    #looping through the json data 
    words = []
    labels = []

    #this is to show what intents they are part off
    docs_x = [] 
    docs_y = []

    #looping through each intent
    for intent in data["intents"]: 

        #get the root of the word and te meaning of the word so as to make the mode more accurate, try to reduce staring the model away from actual points
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)

            #so that way each entry in docs y corrensponds with an entry in docs x
            docs_x.append(wrds)
            docs_y.append(intent["tag"]) 

            if intent["tag"] not in labels:
                labels.append(intent["tag"])


    #stem all the words and remove every duplicate and make sure to check how many words the model has seen
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    #make the words a set so as to remove the duplicates and sort them at the same time
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #one hot encoded to find the frequency of words for the neural network to find out what words are in the string

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    #---------this is where the whole Artificial intelligence process starts from-----------

    #training will be done using numpy first by making them an array and then to train the numpy with the training and also with the output
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f: #writes the file and create it if the file hasnt yet been created. 
        pickle.dump((words, labels, training, output), f)

#build the model using tflearn
tensorflow.compat.v1.reset_default_graph()

#define the input shape expecting for the model as long as the length of training is 0 because each trainig input is the same length
net = tflearn.input_data(shape=[None, len(training[0])])

#this means is to add this fully connected layer to our neural network which starts at input data and have 8 neurons for the hidden layer
#2 hidden layers with 8 neurons that are fully conneced
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #softmax gives probability for each layer of output neurons
net = tflearn.regression(net)

#to train the model
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")

except:
    #pass all of the training data, number of epoch is basically the same time its going to see the same data
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)

    #this is to save the model and this model is used to make some predictions. 
    model.save("model.tflearn")


#to make the predictions and match the input with the model
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)

    return numpy.array(bag)



def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)]) #these are basically probabilities, tries to classify the input into neurons 
        results_index = numpy.argmax(results) #give us the index of the greatest value in the list and then use index to determine repsonse to display
        tag = labels[results_index] #gives the label that it thinks the message us
        

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


chat()