#%%
# classify text into positive or negative sentiment using lstm

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# load data
data = pd.read_csv('reviews_data.csv')

# drop rows with no review text
data = data[data['Review'] != 'No Review Text']
data = data[data['Rating'].notnull()]

# convert ratings to 0 and 1
data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x >= 3 else 0)

# balance the data by taking equal number of positive and negative reviews
pos = data[data['Sentiment'] == 1]
neg = data[data['Sentiment'] == 0]

neg = neg.sample(n=len(pos)+10, random_state=42)
data = pd.concat([pos, neg]) # join positive and negative reviews

# remove punctuation, stop words, lower case them and tokenize
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)
    text = [word for word in text if word not in stop_words]
    return text

data['Review'] = data['Review'].apply(clean_text)

# create a vocabulary of all words
all_words = []
for text in data['Review']:
    for word in text:
        all_words.append(word)

# get unique words
unique_words = set(all_words)

# get length of unique words
voc_len = len(unique_words)

# convert text to sequences
tokenizer = Tokenizer(num_words=voc_len)
tokenizer.fit_on_texts(data['Review'])

sequences = tokenizer.texts_to_sequences(data['Review'])

# get max length of a sequence
max_len = max([len(seq) for seq in sequences])

# pad sequences to max length
padded_sequences = sequence.pad_sequences(sequences, maxlen=max_len)

emb_dim = 64
model = Sequential()
model.add(Embedding(input_dim=voc_len, output_dim=emb_dim, input_length=max_len))
model.add(LSTM(units=100, kernel_regularizer=l2))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# split data into train and test
X = padded_sequences
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#%%
# train model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

model.evaluate(X_test, y_test)