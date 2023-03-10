import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, Flatten, Input
from tensorflow.python.keras.layers import LSTM, Bidirectional, SimpleRNN, Conv1D, MaxPool1D
from keras.engine.topology import Layer, InputSpec
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing import text as keras_text, sequence as keras_seq
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from sklearn.utils import shuffle
from tensorflow import set_random_seed
import gc
import os

#myrand=np.random.randint(1, 99999 + 1)
myrand=58584
np.random.seed(myrand)
set_random_seed(myrand)
z=0

EMBEDDING_SIZE=300
WORDS_SIZE=8000
INPUT_SIZE=700
NUM_CLASSES=6
EPOCHS=6

mydata = pd.read_json('/Users/chempakaseri/Spyder/emotion/Data/anger1k.json')
mydata1 = pd.read_json('/Users/chempakaseri/Spyder/emotion/Data/fear1k.json')
mydata = mydata.append(mydata1)
mydata = shuffle(mydata)

mydata1 = pd.read_json('/Users/chempakaseri/Spyder/emotion/Data/joy1k.json')
mydata = mydata.append(mydata1)
mydata = shuffle(mydata)

mydata1 = pd.read_json('/Users/chempakaseri/Spyder/emotion/Data/love1k.json')
mydata = mydata.append(mydata1)
mydata = shuffle(mydata)

mydata1 = pd.read_json('/Users/chempakaseri/Spyder/emotion/Data/sadness1k.json')
mydata = mydata.append(mydata1)
mydata = shuffle(mydata)

mydata1 = pd.read_json('/Users/chempakaseri/Spyder/emotion/Data/surprise1k.json')
mydata = mydata.append(mydata1)
mydata = shuffle(mydata)

#Change text and emotion to str
mydata['text'] = mydata['text'].astype(str)
mydata = mydata.loc[mydata['emotion'] != "neutral"]
mydata['emotion'] = mydata['emotion'].map({'anger': 0,'fear': 1,'joy': 2,'love': 3,'sadness': 4, 'surprise': 5})
mydata['emotion'] = mydata['emotion'].astype(np.int64)

del(mydata1)
gc.collect()

mydata = shuffle(mydata)
mydata = shuffle(mydata)
mydata = shuffle(mydata)

#   Splitting the data into training (70%) and testing (30$) sets
x_train, x_test, y_train, y_test = train_test_split(mydata.iloc[:,1], mydata.iloc[:,0],
                                                    test_size=0.3, 
                                                    random_state=myrand, 
                                                    shuffle=True)
old_y_test = y_test

#   Prepare tokenizer
##  Create tokkenizer from full list of texts
tokenizer = keras_text.Tokenizer(char_level=False)
tokenizer.fit_on_texts(list(mydata['text']))
tokenizer.num_words=WORDS_SIZE

#   Create sequence file from the tokkenizer for training and testing sets.
## Tokkenizing train data and create matrix
list_tokenized_train = tokenizer.texts_to_sequences(x_train)
x_train = keras_seq.pad_sequences(list_tokenized_train, 
                                  maxlen=INPUT_SIZE,
                                  padding='post')
x_train = x_train.astype(np.int64)

## Tokkenizing test data and create matrix
list_tokenized_test = tokenizer.texts_to_sequences(x_test)
x_test = keras_seq.pad_sequences(list_tokenized_test, 
                                 maxlen=INPUT_SIZE,
                                 padding='post')
x_test = x_test.astype(np.int64)

y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

word2vec = KeyedVectors.load_word2vec_format('/Users/chempakaseri/Spyder/emotion/GoogleNews-vectors-negative300.bin', binary=True)
word_index = tokenizer.word_index

vocabulary_size=min(len(word_index)+1,8000)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_SIZE))
for word, i in word_index.items():
    if i>=WORDS_SIZE:
        continue
    try:
        embedding_vector = word2vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_SIZE)

embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_SIZE,
                            weights=[embedding_matrix],
                            input_length=INPUT_SIZE,
                            trainable=False)

model = Sequential(name='Word2Vec CNN + Bi-RNN + Bi-LSTM')

model.add(embedding_layer)
model.add(Conv1D(filters=250, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=3))
model.add(Conv1D(filters=250, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=3))
model.add(Bidirectional(LSTM(250,return_sequences=True,  activation='relu')))
model.add(Bidirectional(SimpleRNN(250)))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(6, activation='softmax'))

## Define multiple optional optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0, amsgrad=False)

## Compile model with metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Word2Vec CNN + Bi-RNN + Bi-LSTM model built: ")
model.summary()

## Create TensorBoard callbacks

callbackdir= '/Users/chempakaseri/Spyder/emotion/ten'

tbCallback = TensorBoard(log_dir=callbackdir, 
                         histogram_freq=0, 
                         batch_size=128,
                         write_graph=True, 
                         write_grads=True, 
                         write_images=True)

tbCallback.set_model(model)

mld = '/Users/chempakaseri/Spyder/emotion/Models/word2vec_cnn_birnn_bilstm.hdf5'

## Create best model callback
mcp = ModelCheckpoint(filepath=mld, monitor="val_acc",
                      save_best_only=True, mode='max', period=1, verbose=1)

print('Training the Word2Vec CNN + Bi-RNN + Bi-LSTM model')
history = model.fit(x = x_train,
          y = y_train,
          validation_data = (x_test, y_test),
          epochs = EPOCHS,
          batch_size = 128,
          verbose =2,
          callbacks=[mcp,tbCallback])

print('\nConfusion Matrix')
predicted = model.predict_classes(x_test)
confusion = confusion_matrix(y_true=old_y_test, y_pred=predicted)
print(confusion)

## Performance measure
print('\nWeighted Accuracy: '+ str(accuracy_score(y_true=old_y_test, y_pred=predicted)))
print('Weighted precision: '+ str(precision_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted recall: '+ str(recall_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted f-measure: '+ str(f1_score(y_true=old_y_test, y_pred=predicted, average='weighted')))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.plot(epochs_range, acc, 'bo', label='Training acc')
plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()