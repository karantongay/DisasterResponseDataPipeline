import sys

import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle as pkl


def load_data(database_filepath):
    """
    This function reads the data from database where the clean
    data is stored.
    Note: It is observed that the 'child_alone' category in the
    data had only one class which was '0' and therefore, at the
    time of loading the data, we are currently deleting that
    category attribute. Although, to keep it scalable, the
    attribute still persists in the database, just in case if new 
    data comes in the future that is classified as child_alone.
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('figure-eight', engine)
    # There is only one classification class for child_alone which is 0 which indicates that there is no message classified into this class.
    del df['child_alone']
    X = df.message.values
    Y = df[np.delete(df.columns.values, [0,1,2,3])].values
    category_names = np.delete(df.columns.values, [0,1,2,3])
    return X,Y,category_names


def tokenize(text):
    """
    This function tokenizes the input text and performs necessary cleaning.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    This function builds a pipeline model which makes use of:
    a. CountVectorizer()
    b. TfidfTransformer()
    c. MultiOutputClassifier(SGDClassifier())
    
    Note: The parameters used to call the above steps in the
    pipeline were chosen after performing validation tests
    using multiple parameters with GridSearchCV.
    
    parameters_SGD = {
     'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
     'tfidf__use_idf': (True, False),
     'clf__estimator__alpha': (1e-2, 1e-3, 1e-4, 1e-1),
    }

    cv = GridSearchCV(model, param_grid=parameters_SGD, n_jobs = -1)
    cv.get_params().keys()
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    
    SGDClassifier is chosen after testing and tuning other
    classfiers namely, RandomForest, NaiveBayes and k-NN.
    
    """
    text_clf_SGD = Pipeline([
        ('vect', CountVectorizer(ngram_range = (1,2))),
         ('tfidf', TfidfTransformer(use_idf = True)),
         ('clf', MultiOutputClassifier(SGDClassifier()))
        ])

    parameters_SGD = {
        'vect__ngram_range': [(1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__estimator__alpha': (1e-2, 1e-3, 1e-4, 1e-1),
        }
        
    model = GridSearchCV(text_clf_SGD, param_grid=parameters_SGD, n_jobs = -1, verbose=3, cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the new model and generates classification
    report containing precision, recall, f-score and accuracy information
    for individual classes.
    """
    y_pred = model.predict(X_test)
    for x in range(0, len(category_names)):
        print(category_names[x])
        print(classification_report(Y_test[:,x], y_pred[:,x]))
        print("Accuracy: " + str(accuracy_score(Y_test[:, x], y_pred[:, x])))


def save_model(model, model_filepath):
    """
    This function packages the trained model into the pickle file.
    """
    # save the classifier
    with open(model_filepath, 'wb') as fid:
        pkl.dump(model, fid)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()