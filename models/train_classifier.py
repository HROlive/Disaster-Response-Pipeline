"""
Training Classifier

Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disaster_response.db classifier.pkl

Arguments Description:
    1) Path to SQLite database containing pre-processed data (e.g. disaster_response.db)
    2) Path to pickle file where the model will be saved (e.g. classifier.pkl)
"""

# import necessary libraries
import pandas as pd
import numpy as np
import os
import sys
import pickle
import re
import nltk
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath):
    # load data from the database
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name, engine)
    # split the features and labels
    X = df['message']
    Y = df.iloc[:,4:]
    # get the column names
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Detect all the urls from the text 
    detected_urls = re.findall(url_regex, text)
    # Replace all urls with a placeholder string
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize, normalize case and remove whitespaces
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]
    
    return clean_tokens
    
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(xgb.XGBClassifier(learning_rate=0.02, objective='binary:hinge')))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_child_weight': [1, 5, 10],
        'clf__estimator__gamma': [0.5, 1, 1.5, 2, 5],
        'clf__estimator__subsample': [0.6, 0.8, 1.0],
        'clf__estimator__colsample_bytree': [0.6, 0.8, 1.0],
        'clf__estimator__max_depth': [3, 4, 5]
        }
        
    model = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=1,
                               scoring='roc_auc', n_jobs=-1, cv=5, verbose=3, random_state=42)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print('Classification Report: ')
    print(classification_report(Y_pred, Y_test.values, target_names=category_names))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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