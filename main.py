import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def text_preprocessing(csv_file, random_sample):
    # 1. Exploratory Data Analysis
    print('Basic Info From', csv_file, 'Data')
    df = pd.read_csv(csv_file)

    if random_sample:
        df = df.sample(frac=.01, random_state=39)

    basic_info = df.describe()  # get basic info such as count and mean
    print(basic_info)  # mean of .237 showing more negative than positive

    sns.histplot(data=df['Sentiment'])
    plt.title(str(csv_file))
    plt.show()  # visual representation of data showing skew towards negative

    print('Missing Data')
    print(df.isna().sum())  # check for null values and find that there are none

    # 2. Text Preprocessing
    # 2.1 Lower Case
    df['Text'] = df['Text'].str.lower()
    print('\nResult from Step 1 - Lower Casing')
    print(df.head()['Text'])

    # 2.2 Remove Digital Numbers
    import re

    df['Text'] = df['Text'].apply(
        lambda text: re.sub('[0-9]', '', text))  # replace all digital numbers with white space
    print('\nResult from Step 2 - Remove Digital Numbers')
    print(df.head()['Text'])

    # 2.3 Remove Contractions
    cList = {
        "ain\'t": "am not",
        "aren\'t": "are not",
        "can\'t": "cannot",
        "can\'t've": "cannot have",
        "\'cause": "because",
        "could\'ve": "could have",
        "couldn\'t": "could not",
        "couldn\'t've": "could not have",
        "didn\'t": "did not",
        "doesn\'t": "does not",
        "don\'t": "do not",
        "hadn\'t": "had not",
        "hadn\'t\'ve": "had not have",
        "hasn\'t": "has not",
        "haven\'t": "have not",
        "he\'d": "he would",
        "he\'d\'ve": "he would have",
        "he\'ll": "he will",
        "he\'ll\'ve": "he will have",
        "he\'s": "he is",
        "how\'d": "how did",
        "how\'d\'y": "how do you",
        "how\'ll": "how will",
        "how\'s": "how is",
        "i\'d": "i would",
        "i\'d'\ve": "i would have",
        "i\'ll": "i will",
        "i\'ll\'ve": "i will have",
        "i\'m": "i am",
        "i\'ve": "i have",
        "isn\'t": "is not",
        "it\'d": "it had",
        "it\'d\'ve": "it would have",
        "it\'ll": "it will",
        "it\'ll\'ve": "it will have",
        "it\'s": "it is",
        "let\'s": "let us",
        "ma\'am": "madam",
        "mayn\'t": "may not",
        "might\'ve": "might have",
        "mightn\'t": "might not",
        "mightn\'t\'ve": "might not have",
        "must\'ve": "must have",
        "mustn\'t": "must not",
        "mustn\'t\'ve": "must not have",
        "needn\'t": "need not",
        "needn\'t\'ve": "need not have",
        "o\'clock": "of the clock",
        "oughtn\'t": "ought not",
        "oughtn\'t\'ve": "ought not have",
        "shan\'t": "shall not",
        "sha\'n\'t": "shall not",
        "shan\'t\'ve": "shall not have",
        "she\'d": "she would",
        "she\'d\'ve": "she would have",
        "she\'ll": "she will",
        "she\'ll\'ve": "she will have",
        "she\'s": "she is",
        "should\'ve": "should have",
        "shouldn\'t": "should not",
        "shouldn\'t\'ve": "should not have",
        "so\'ve": "so have",
        "so\'s": "so is",
        "that\'d": "that would",
        "that\'d\'ve": "that would have",
        "that\'s": "that is",
        "there\'d": "there had",
        "there\'d've": "there would have",
        "there\'s": "there is",
        "they\'d": "they would",
        "they\'d\'ve": "they would have",
        "they\'ll": "they will",
        "they\'ll\'ve": "they will have",
        "they\'re": "they are",
        "they\'ve": "they have",
        "to\'ve": "to have",
        "wasn\'t": "was not",
        "we\'d": "we had",
        "we\'d\'ve": "we would have",
        "we\'ll": "we will",
        "we\'ll\'ve": "we will have",
        "we\'re": "we are",
        "we\'ve": "we have",
        "weren\'t": "were not",
        "what\'ll": "what will",
        "what\'ll\'ve": "what will have",
        "what\'re": "what are",
        "what\'s": "what is",
        "what\'ve": "what have",
        "when\'s": "when is",
        "when\'ve": "when have",
        "where\'d": "where did",
        "where\'s": "where is",
        "where\'ve": "where have",
        "who\'ll": "who will",
        "who\'ll\'ve": "who will have",
        "who\'s": "who is",
        "who\'ve": "who have",
        "why\'s": "why is",
        "why\'ve": "why have",
        "will\'ve": "will have",
        "won\'t": "will not",
        "won\'t\'ve": "will not have",
        "would\'ve": "would have",
        "wouldn\'t": "would not",
        "wouldn\'t've": "would not have",
        "y\'all": "you all",
        "y\'alls": "you alls",
        "y\'all\'d": "you all would",
        "y\'all\'d\'ve": "you all would have",
        "y\'all\'re": "you all are",
        "y\'all\'ve": "you all have",
        "you\'d": "you had",
        "you\'d\'ve": "you would have",
        "you\'ll": "you you will",
        "you\'ll\'ve": "you you will have",
        "you\'re": "you are",
        "you\'ve": "you have"
    }  # contractions list taken from https://gist.github.com/nealrs/96342d8231b75cf4bb82

    def convert_contractions(text):
        for key in cList:
            text = re.sub(str(key), str(cList[key]), text)  # loop through each contraction and add when needed
        return text

    df['Text'] = df['Text'].apply(lambda text: convert_contractions(text))  # convert contractions fo ease of use
    print('\nResult from Step 3 - Remove Contractions')
    print(df.head()['Text'])

    # 2.4 Remove URLS
    df['Text'] = df['Text'].apply(lambda text: re.sub(r'http\S+', ' ', text))  # remove anything starting with http
    print('\nResult from Step 4 - Remove URLS')
    print(df.head()['Text'])

    # 2.5 Remove UserNames
    df['Text'] = df['Text'].apply(lambda text: re.sub('@\S+', ' ', text))  # remove anything starting with an @
    print('\nResult from Step 5 - Remove UserNames')
    print(df.head()['Text'])

    # 2.6 Removing Special Characters, Punctuation, and Symbols
    df['Text'] = df['Text'].apply(lambda text: re.sub('[^a-z0-9<>]', ' ', text))
    print('\nResult from Step 6 - Remove Special Characters, Punctuation, and Symbols')
    print(df.head()['Text'])

    # 2.7 Sentence Segmentation
    import nltk
    df['Text'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split()]))
    print('\nResult from Step 7 - Sentence Tokenization')
    print(df.head()['Text'], '\n')

    # 2.8 Lemmatize
    from nltk.stem import WordNetLemmatizer

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define a function to lemmatize a sentence
    def lemmatize_sentence(sentence):
        words = nltk.word_tokenize(sentence)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    # Apply the lemmatize_sentence function to each sentence in the CSV file
    df['Text'] = df['Text'].apply(lemmatize_sentence)

    print('2.8 Lemmatized Words')
    print(df.head()['Text'])
    return df  # return the clean dataframe


def print_metrics(accuracy, report, roc_auc):
    print('Accuracy:', accuracy)
    print(report)
    print('AUC-ROC:', roc_auc)
    print()


# main
print('Beginning of Text Analysis')

cleanTest = text_preprocessing('test.csv', random_sample=False)
cleanTrain = text_preprocessing('train.csv', random_sample=True)


# Feature Extraction

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Bag of Words
vectorizer = CountVectorizer()
train_bag = vectorizer.fit_transform(cleanTrain['Text'])  # fit_transform to get the vocab from train
test_bag = vectorizer.transform(cleanTest['Text'])  # transform around larger dfs vocab

# Logistic Regression
lc = LogisticRegression()
lc.fit(train_bag.toarray(), cleanTrain['Sentiment'])  # toarray() to make dense matrix

bow_predictions = lc.predict(test_bag)  # get predictions with test data
bow_accuracy = accuracy_score(cleanTest['Sentiment'], bow_predictions)
bow_report = classification_report(cleanTest['Sentiment'], bow_predictions)

# Calculate AUC-ROC for BoW predictions
bow_roc_auc = roc_auc_score(cleanTest['Sentiment'], lc.predict_proba(test_bag)[:, 1])

# Print the metrics
print('\n', 'LC Bag of Words Metrics: ', '\n')
print_metrics(bow_accuracy, bow_report, bow_roc_auc)

# SVC
svc = SVC(probability=True, max_iter=100)
svc.fit(train_bag, cleanTrain['Sentiment'])  # toarray() to make dense matrix

bow_predictions = svc.predict(test_bag)  # get predictions with test data
bow_accuracy = accuracy_score(cleanTest['Sentiment'], bow_predictions)
bow_report = classification_report(cleanTest['Sentiment'], bow_predictions)

# Calculate AUC-ROC for BoW predictions
bow_roc_auc = roc_auc_score(cleanTest['Sentiment'], svc.predict_proba(test_bag)[:, 1])

# Print the metrics
print('\n', 'SVC Bag of Words Metrics: ', '\n')
print_metrics(bow_accuracy, bow_report, bow_roc_auc)

# NB
nbc = GaussianNB()
nbc.fit(train_bag.toarray(), cleanTrain['Sentiment'])  # toarray() to make dense matrix

bow_predictions = nbc.predict(test_bag.toarray())  # get predictions with test data
bow_accuracy = accuracy_score(cleanTest['Sentiment'], bow_predictions)
bow_report = classification_report(cleanTest['Sentiment'], bow_predictions)

# Calculate AUC-ROC for BoW predictions
bow_roc_auc = roc_auc_score(cleanTest['Sentiment'], nbc.predict_proba(test_bag.toarray())[:, 1])

# Print the metrics
print('\n', 'NBC Bag of Words Metrics: ', '\n')
print_metrics(bow_accuracy, bow_report, bow_roc_auc)

# Random Forest
rfc = RandomForestClassifier()
rfc.fit(train_bag.toarray(), cleanTrain['Sentiment'])  # toarray() to make dense matrix

bow_predictions = rfc.predict(test_bag)  # get predictions with test data
bow_accuracy = accuracy_score(cleanTest['Sentiment'], bow_predictions)
bow_report = classification_report(cleanTest['Sentiment'], bow_predictions)

# Calculate AUC-ROC for BoW predictions
bow_roc_auc = roc_auc_score(cleanTest['Sentiment'], rfc.predict_proba(test_bag)[:, 1])

# Print the metrics
print('\n', 'RFC Bag of Words Metrics: ', '\n')
print_metrics(bow_accuracy, bow_report, bow_roc_auc)

# Extracting TF-IDF parameters
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
train_tfidf = tfidf_vectorizer.fit_transform(cleanTrain['Text'])  # create around train
test_tfidf = tfidf_vectorizer.transform(cleanTest['Text'])  # transform test

# Logistic Regression
lc.fit(train_tfidf.toarray(), cleanTrain['Sentiment'])
tfidf_predictions = lc.predict(test_tfidf)
tfidf_accuracy = accuracy_score(cleanTest['Sentiment'], tfidf_predictions)
tfidf_report = classification_report(cleanTest['Sentiment'], tfidf_predictions)

# Calculate AUC-ROC for tf-idf predictions
tfidf_roc_auc = roc_auc_score(cleanTest['Sentiment'], lc.predict_proba(test_tfidf)[:, 1])

print('LC tf-idf Metrics:')
print_metrics(tfidf_accuracy, tfidf_report, tfidf_roc_auc)

# SVC
svc.fit(train_tfidf.toarray(), cleanTrain['Sentiment'])
tfidf_predictions = svc.predict(test_tfidf.toarray())
tfidf_accuracy = accuracy_score(cleanTest['Sentiment'], tfidf_predictions)
tfidf_report = classification_report(cleanTest['Sentiment'], tfidf_predictions)

# Calculate AUC-ROC for tf-idf predictions
tfidf_roc_auc = roc_auc_score(cleanTest['Sentiment'], svc.predict_proba(test_tfidf.toarray())[:, 1])

print('SVC tf-idf Metrics:')
print_metrics(tfidf_accuracy, tfidf_report, tfidf_roc_auc)

# NBC
nbc.fit(train_tfidf.toarray(), cleanTrain['Sentiment'])
tfidf_predictions = nbc.predict(test_tfidf.toarray())
tfidf_accuracy = accuracy_score(cleanTest['Sentiment'], tfidf_predictions)
tfidf_report = classification_report(cleanTest['Sentiment'], tfidf_predictions)

# Calculate AUC-ROC for tf-idf predictions
tfidf_roc_auc = roc_auc_score(cleanTest['Sentiment'], nbc.predict_proba(test_tfidf.toarray())[:, 1])

print('NBC tf-idf Metrics:')
print_metrics(tfidf_accuracy, tfidf_report, tfidf_roc_auc)

# RFC
rfc.fit(train_tfidf.toarray(), cleanTrain['Sentiment'])
tfidf_predictions = rfc.predict(test_tfidf.toarray())
tfidf_accuracy = accuracy_score(cleanTest['Sentiment'], tfidf_predictions)
tfidf_report = classification_report(cleanTest['Sentiment'], tfidf_predictions)

# Calculate AUC-ROC for tf-idf predictions
tfidf_roc_auc = roc_auc_score(cleanTest['Sentiment'], rfc.predict_proba(test_tfidf.toarray())[:, 1])

print('RFC tf-idf Metrics:')
print_metrics(tfidf_accuracy, tfidf_report, tfidf_roc_auc)

import numpy as np
# Word2vec feature extraction
from gensim.models import Word2Vec

sentences = [sentence.split() for sentence in cleanTrain['Text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
# using size 100 as mentioned in class


def vectorize(sentence):  # takes in a sentence from list of sentences
    words = sentence.split()  # split into words
    # create word vecs if word is in w2v vocab
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(100)  # create empty array of zeros so they're both the same size
    words_vecs = np.array(words_vecs)  # make into dense np array
    return words_vecs.mean(axis=0)  # return


train_w2v = np.array([vectorize(sentence) for sentence in cleanTrain['Text']])   # vectorize each sentence
test_w2v = np.array([vectorize(sentence) for sentence in cleanTest['Text']])  # vectorize each sentence

lc.fit(train_w2v, cleanTrain['Sentiment'])  # fit to logisitc regression

w2v_predictions = lc.predict(test_w2v)
w2v_accuracy = accuracy_score(cleanTest['Sentiment'], w2v_predictions)
w2v_report = classification_report(cleanTest['Sentiment'], w2v_predictions)

# Calculate AUC-ROC for Word2vec predictions
w2v_roc_auc = roc_auc_score(cleanTest['Sentiment'], lc.predict_proba(test_w2v)[:, 1])

print('LC Word2vec Metrics:')
print_metrics(w2v_accuracy, w2v_report, w2v_roc_auc)

# SVC
svc.fit(train_w2v, cleanTrain['Sentiment'])  # fit to svc

w2v_predictions = svc.predict(test_w2v)
w2v_accuracy = accuracy_score(cleanTest['Sentiment'], w2v_predictions)
w2v_report = classification_report(cleanTest['Sentiment'], w2v_predictions)

# Calculate AUC-ROC for Word2vec predictions
w2v_roc_auc = roc_auc_score(cleanTest['Sentiment'], svc.predict_proba(test_w2v)[:, 1])

print('SVC Word2vec Metrics:')
print_metrics(w2v_accuracy, w2v_report, w2v_roc_auc)

# NBC
nbc.fit(train_w2v, cleanTrain['Sentiment'])  # fit to nbc

w2v_predictions = nbc.predict(test_w2v)
w2v_accuracy = accuracy_score(cleanTest['Sentiment'], w2v_predictions)
w2v_report = classification_report(cleanTest['Sentiment'], w2v_predictions)

# Calculate AUC-ROC for Word2vec predictions
w2v_roc_auc = roc_auc_score(cleanTest['Sentiment'], nbc.predict_proba(test_w2v)[:, 1])

print('NBC Word2vec Metrics:')
print_metrics(w2v_accuracy, w2v_report, w2v_roc_auc)

# RFC
rfc.fit(train_w2v, cleanTrain['Sentiment']) # fit to rfc
w2v_predictions = rfc.predict(test_w2v)
w2v_accuracy = accuracy_score(cleanTest['Sentiment'], w2v_predictions)
w2v_report = classification_report(cleanTest['Sentiment'], w2v_predictions)

# Calculate AUC-ROC for Word2vec predictions
w2v_roc_auc = roc_auc_score(cleanTest['Sentiment'], rfc.predict_proba(test_w2v)[:, 1])

print('RFC Word2vec Metrics:')
print_metrics(w2v_accuracy, w2v_report, w2v_roc_auc)

print('End of Text Analysis')
