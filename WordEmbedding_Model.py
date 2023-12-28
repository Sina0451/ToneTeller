import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
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

# Block for playing with tokens and words
"""
for key, value in vectorizer.vocabulary_.items():
    # value ==> Token of the word
    # key ==> The word ITSELF
    print('** {} ==> {}'.format(value, key))
"""


# ---------------------------------------------------------------------------------------------------------------

# **** TOKENIZER FUNCTIONS ****

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print("THE SENTENCE:", sentences_train[2])
print(" TOKENIZED:", X_train[2])

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print("SEQUENCED WORD:", X_train[0])

# ---------------------------------------------------------------------------------------------------------------

# **** MAIN PART OF THE PROGRAM; THE MODEL &  THE STRUCTURE

input_dim = X_train.shape[1]

# noinspection PyBroadException
try:
    model = load_model('Model Weights/WordEmbedding_Model.h5')
except:

    embedding_dim = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.9))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.9))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                        epochs=30,
                        verbose=True,
                        validation_data=(X_test, y_test),
                        batch_size=10)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)

    model.save('Model Weights/WordEmbedding_Model.h5')

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

    X_test = tokenizer.texts_to_sequences(input_test)
    print(X_test)
    vocab_size = len(tokenizer.word_index)
    maxlen = 100
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    print(X_test)
    print(model.predict(X_test))
