import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
import pickle

## loading english stopwords
stopwords = set(stopwords.words('english'))


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages',engine)
    X = df['message']
    Y = df.iloc[:,4:]

    return X,Y,Y.columns.tolist()

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    pipeline = Pipeline([('tfidf',TfidfVectorizer(tokenizer=tokenize)),
                     ('clf',MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'clf__estimator__n_estimators':[20],'clf__estimator__max_depth':[3], 
              'clf__estimator__min_samples_split':[2]}

    cv = GridSearchCV(pipeline,param_grid=parameters,scoring='f1_weighted',verbose=1,n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    predictions = model.predict(X_test)
    for col_no in range(predictions.shape[1]):
        break_str = '-'*20 + "Precision Recall Report for " + category_names[col_no] + "-"*20
        print(break_str)
        print(str(classification_report(Y_test.values[:,col_no],predictions[:,col_no])))



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath,'wb'))


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

        ## best model
        model = model.best_estimator_
        
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