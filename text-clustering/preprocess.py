import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tokenization import create_alphanum_tokenizer, WordTokenizer
from file_utils import create_directory, read_stop_words_file

def vaccination_tweets(file_path):
    stop_words = stopwords.words('english')
    # add dataset specific stop words
    stop_words.extend(read_stop_words_file('vaccination_tweets_stop_words.txt'))

    word_tokenizer = WordTokenizer(tokenizer=create_alphanum_tokenizer(), stopwords=stop_words, lemmatizer=WordNetLemmatizer(), remove_digits=False)
    
    df = pd.read_csv(file_path)
    
    # to remove URL before tokenization
    url_pattern = re.compile(r'https?://\S+')
    df['text'] = df.apply(lambda row: url_pattern.sub('',row['text']), axis=1)
    
    # to remove usernames @*
    username_pattern = re.compile(r'@\S+')
    df['text'] = df.apply(lambda row: username_pattern.sub('',row['text']), axis=1)
    
    # to remove hashtag #*
    hashtag_pattern = re.compile(r'#\S+')
    df['text'] = df.apply(lambda row: hashtag_pattern.sub('',row['text']), axis=1)
    
    df['tokens'] = df.apply(lambda row: word_tokenizer.tokenize(row['text']), axis=1)

    # remove rows with no tokens
    df = df[df['tokens'].str.len() > 0]

    return df

if __name__ == "__main__":

    DOWNLOAD_DIR = './dataset'
    TEXT_DATA_DIR = './text/'
    
    create_directory(TEXT_DATA_DIR)
    
    nltk.download('stopwords', download_dir='/tmp')
    
    # vaccination_tweets
    file_path = '{}/vaccination_tweets.csv'.format(DOWNLOAD_DIR)
    vaccination_tweets(file_path).to_pickle('{}/vaccination_tweets.pkl'.format(TEXT_DATA_DIR))