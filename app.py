import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import  pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout,GRU,Bidirectional

#Load the dataset
data = gutenberg.raw('shakespeare-hamlet.txt')

## Tokenize the text-creating indexes for words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index)+1

## create inoput sequences
input_sequences = []
for line in data.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

## Pad Sequences
max_sequence_len=max([len(x) for x in input_sequences])
input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))

##create predicitors and label
x,y=input_sequences[:,:-1],input_sequences[:,-1]
y=tf.keras.utils.to_categorical(y,num_classes=total_words)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



# Define the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))

# Adding Bidirectional GRU layer
model.add(Bidirectional(GRU(150, return_sequences=True)))
model.add(Dropout(0.2))

# Adding another Bidirectional GRU layer
model.add(Bidirectional(GRU(100, return_sequences=True)))
model.add(Dropout(0.2))

# Adding a regular GRU layer
model.add(GRU(100))
model.add(Dropout(0.2))

# Adding a Dense layer for classification
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

# Output layer for multi-class classification
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Train the model
history=model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),verbose=1)
