# Load and stopwords
import pandas as pd
# Train test split
from sklearn.model_selection import train_test_split
# Text pre-processing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('dataset.csv')
# rename the columns
df.rename(columns={'v1': 'text', 'v2': 'humor'}, inplace=True)
print(df.head())

# get length column
df['text_length'] = df['text'].apply(len)

# convert humor label to numeric value
df['is_humor'] = df["humor"].astype(int)

# the longest text length is 99 characters
# print(df['text_length'].max())

# split the data into train and test -- I am using a 80/20 split but can be changed
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['is_humor'], test_size=0.2, random_state=999)

# Defining pre-processing parameters
trunc_type = 'post'
padding_type = 'post'
# out of vocabulary token
oov_tok = '<OOV>'

# maximum number of unique tokens
vocab_size = 2000

tokenizer = Tokenizer(num_words=vocab_size,
                      char_level=False,
                      oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)

# padding the train data
training_sequences = tokenizer.texts_to_sequences(x_train)
training_padded = pad_sequences(training_sequences, maxlen=100,
                                padding=padding_type, truncating=trunc_type)

# padding the test data
testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences, maxlen=100,
                               padding=padding_type, truncating=trunc_type)

# below is the shape of the padded tokenized data
print('Shape of training tensor: ', training_padded.shape)
print('Shape of testing tensor: ', testing_padded.shape)