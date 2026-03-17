# This is the original file that I used to test out the logistic model.
# It is not used in the final project, but was used in the early stages of development to specifically test the best text representation.
# Note that a loto f hte code has been commented out since I was constantly changing this file to test things.

import kagglehub
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gensim.downloader
from gensim.models import KeyedVectors
import re
from sklearn.feature_extraction.text import CountVectorizer
# Download latest version
path = "training_set_rel3.tsv"
path2 = kagglehub.dataset_download("lburleigh/asap-2-0")
nltk.download('stopwords')
nltk.download('punkt')
# print("Path to dataset files:", path)
# data_array = pd.read_csv(path + '\\ASAP2_train_sourcetexts.csv')

data_array = pd.read_csv(path, sep = '\t', encoding='latin1')
stemmer = PorterStemmer()

def process_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,\']', ' ', text)
    tokens = word_tokenize(text)
    stopwords_set = set(stopwords.words('english'))
    # tokens = [stemmer.stem(word) for word in tokens if word not in stopwords_set]
    tokens = [word for word in tokens if word not in stopwords_set]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


preprocess_data = lambda text : process_text(text)
data_array["essay"] = data_array["essay"].apply(preprocess_data)
essay_list = data_array['essay']

# Y = data_array['domain1_score'] # unsure if this is a good metric or not, since this combines the scores of all raters.
print(essay_list)
Y = data_array['rater1_domain1']
np.random.seed(42)
model = LogisticRegression(l1_ratio=0, max_iter=200)
document_number = 0
# uncomment once, then recomment and load from KeyedVectors
# vector_model = gensim.downloader.load("glove-wiki-gigaword-300")
# vector_model.save('pretrained')
vector_model = KeyedVectors.load('pretrained', mmap = 'r')
def sentence_vectorizer(sentence, model):
    global document_number
    document_number += 1
    # print(f"document {document_number} processed")
    words = sentence.split(" ")
    words = [w for w in words if w in model]
    sentence_vector = np.array([model[word] for word in words])
    return sentence_vector
X = [sentence_vectorizer(s, vector_model) for s in essay_list]
vectorizer = CountVectorizer()
X2 = vectorizer.fit_transform(essay_list)
frequency_matrix = pd.DataFrame(
    X2.toarray(), 
    columns=vectorizer.get_feature_names_out()
)
print(frequency_matrix)

def fix_rows_np(rows, fill_value=0):
    # Compute max row length
    max_len = max(len(r) for r in rows)

    # Create full array filled with fill_value
    out = np.full((len(rows), max_len), fill_value, dtype=object)

    # Copy data into array
    for i, r in enumerate(rows):
        for j in range(len(r)):
            out[i][j] = r[j][0]
    return out

X = fix_rows_np(X)
print(X.shape)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.5, random_state=42)
model.fit(X_train, y_train)

def evaluate_model(model, X, y_true):
    y_predict = model.predict(X)
    print(f"Accuracy score: {accuracy_score(y_true, y_predict)}")
    print(f"Mean absolute error: {mean_absolute_error(y_true, y_predict)}")
    print(f"Mean square error: {mean_squared_error(y_true, y_predict)}")


evaluate_model(model, X_train, y_train)
evaluate_model(model, X_val, y_val)