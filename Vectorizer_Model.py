import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------------------------------------------

# **** PLOT FUNCTIONS ****

# Draws charts for training and validation details
plt.style.use('ggplot')


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------------------------------------------

# **** DATA FOLDER (COMMENTS ON YELP, AMAZON, IMDB) OPENING AND READING FUNCTIONS ****

filepath_dict = {'yelp': 'Data/sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': 'Data/sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb': 'Data/sentiment labelled sentences/imdb_labelled.txt'}

df_list = []
# Reading and merging All databases into one
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

# df --> All databases merged and sorted by Source, Sentences Values, Label
df = pd.concat(df_list)

# Complete Info of df (whole database)
# print(df.info())

df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values  # len --> 1000
y = df_yelp['label'].values

test = df

test_x = test['sentence'].values
test_y = test['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

# ---------------------------------------------------------------------------------------------------------------

# **** VECTORING THE WORDS ****

# Building the tokenizer as "vectorizer"
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

vectorizer.fit(test_x)

# print (vectorizer.vocabulary_.items())

# Information of an element in database
# print((df_yelp.iloc[0]))

# Tokenizing the train and test databases
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

predict_inp = vectorizer.transform(test_x)

# Block for playing with tokens and words
"""
for key, value in vectorizer.vocabulary_.items():
    # value ==> Token of the word
    # key ==> The word ITSELF
    print('** {} ==> {}'.format(value, key))
"""


# ---------------------------------------------------------------------------------------------------------------

# **** MAIN PART OF THE PROGRAM; THE MODEL &  THE STRUCTURE

input_dim = X_train.shape[1]

# noinspection PyBroadException
try:
    model = load_model('Model Weights/Vectorizer_Model.h5')
except:

    # Building up the model :)
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))  # Params --> (10 * 1714) + (10 * 1) = 17150
    model.add(layers.Dense(1, activation='relu'))  # Params --> 10 (previous outputs) + 1 = 11

    # Rebuild and Reset the model and weights
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Gives a summary about Neural Network
    model.summary()

    history = model.fit(X_train, y_train,
                        epochs=15,
                        verbose=2,
                        validation_data=(X_test, y_test),
                        batch_size=10)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    # noinspection PyRedeclaration
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)

    model.save('Model Weights/Vectorizer_Model.h5')

print('------------------------------------------------------------------------------------')

# Checking the predicted word to see if they are Correct or Wrong
"""
output = (model.predict(predict_inp))
output_int = []

for i in range(len(output)):
    if (output[i] > 0.5):   output_int.append(1)
    else:    output_int.append(0)

for i in range (len(output)):
    if (output_int[i] == test_y[i]):
        print ("CORRECT", test_x[i], "label:", test_y[i], "output:", output[i])
    else:
        print ("**** WRONG", test_x[i], "label:", test_y[i], "output:", output[i], "int:", output_int[i])
"""

# ---------------------------------------------------------------------------------------------------------------

# **** MANUAL INPUTS AND VISUAL OUTPUT OF THE MODEL ****

while True:
    input_test = input('>>> ')
    input_test = np.array([input_test])

    print(input_test)

    X_test = vectorizer.transform(input_test)
    print(X_test)

    print(model.predict(X_test))
