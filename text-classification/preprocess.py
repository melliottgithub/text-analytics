import os
import pandas as pd
import nltk
# from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(directory, subset):
    news = fetch_20newsgroups(data_home=directory, remove=('headers', 'footers', 'quotes'),
                              subset=subset, shuffle=True, random_state=42, download_if_missing=True)
    return news

def tokenizer(text, stop_words, lemmatizer):
    alphanum_tokenize = RegexpTokenizer(r"[a-zA-Z0-9-]+")
    tokens = alphanum_tokenize.tokenize(text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if len(token) > 1 and token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def create_dataframe(subset, tokenizer, stop_words, lemmatizer):
    columns = {
        'tokens': list(tokenizer(text, stop_words, lemmatizer) for text in subset['data']),
        'categoryid':  subset['target']
    }
    df = pd.DataFrame(data=columns)
    return df

if __name__ == "__main__":
    
    nltk.download('stopwords', download_dir='/tmp')
    stop_words = stopwords.words('english')
    stop_words.append('--')
    
    lemmatizer = WordNetLemmatizer()
    
    DOWNLOAD_DIR = './dataset'
    NEWSGROUPS_DATA_DIR = './text/newsgroups'
    
    create_directory(NEWSGROUPS_DATA_DIR)
    
    # download datasets
    train = load_dataset(DOWNLOAD_DIR, 'train')
    print('Train data: downloaded', len(train['data']), 'items')
    
    test = load_dataset(DOWNLOAD_DIR, 'test')
    print('Test data: downloaded', len(test['data']), 'items')

    # clean-up and tokenization
    
    train_df = create_dataframe(train, tokenizer, stop_words, lemmatizer)
    train_df.to_pickle('{}/train.pkl'.format(NEWSGROUPS_DATA_DIR))
    
    test_df = create_dataframe(test, tokenizer, stop_words, lemmatizer)
    test_df.to_pickle('{}/test.pkl'.format(NEWSGROUPS_DATA_DIR))
    
    # save categories id and labels
    columns = {
        'id':  list(range(len(train['target_names']))),
        'name': train['target_names']
    }
    cat_df = pd.DataFrame(data=columns)
    cat_df.to_pickle('{}/categories.pkl'.format(NEWSGROUPS_DATA_DIR))