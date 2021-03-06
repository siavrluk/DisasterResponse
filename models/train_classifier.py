import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import pickle



def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM messages_categories_clean' ,engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    column_names = Y.columns
    return X, Y, column_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
#        'vect__max_df': (0.5, 1.0),
#        'vect__max_features': (None, 5000, 10000),
#        'tfidf__use_idf': (True, False),
        'clf__estimator__criterion': ['gini'],
        'clf__estimator__max_depth': [2, 5],
        'clf__estimator__n_estimators': [10, 20, 50]
#         'clf__estimator__min_samples_leaf':[1, 5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    
    
    pipeline2 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('chi2', SelectKBest(chi2, k=500)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    
    parameters2 = {
        'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
       'chi2__k': [500, 1000],
       'clf__estimator__n_estimators': [100, 200]
    }

    cv2 = GridSearchCV(pipeline2, param_grid=parameters2)
    
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    for j in range(len(Y_test.columns)):
        print(Y_test.columns[j])
        print(classification_report(Y_pred[:,j], Y_test.values[:,j]))

def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))


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