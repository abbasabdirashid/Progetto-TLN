import string
from pprint import pprint

import gensim
import pandas as pd
from gensim.models import LdaModel
from gensim.test.utils import common_texts

from gensim.corpora.dictionary import Dictionary


from nltk import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()


# import plotly.express as px
# import plotly.graph_objects as go


FILE_PATH='utils/story.txt'


def open_file(path):
    file=open(path,'r')
    text=file.read()
    file.close()
    pars=text.split('/n')
    return pars


def clean_sentence(line):
    stop_words = set(stopwords.words('english'))
    puncts = ['', '‘', '’', '”', '“','–']

    words = word_tokenize(line)
    filtered_sentence = [w.lower().strip(string.punctuation) for w in words if
                         not w.lower().strip(string.punctuation) in stop_words and w.lower().strip(
                             string.punctuation) not in puncts and not w.isnumeric()]
    return filtered_sentence

def get_arrs_pars(pars):
    arrs=[]
    for par in pars:
        arrs.append(clean_sentence(par))
    return arrs

def extract_topics(texts,numTopics):
    # we want to associate each word in the corpus with a unique integer ID.
    # We can do this using the gensim.corpora.Dictionary class.
    # This dictionary defines the vocabulary of all words that our processing knows about.
    dictionary = Dictionary(texts)


    # We can create the bag-of-word representation for a document using the doc2bow method of the dictionary,
    # which returns a sparse representation of the word counts
    corpus = [dictionary.doc2bow(text) for text in texts]


    # Build LDA model
    lda_model = LdaModel(corpus=corpus,id2word=dictionary,num_topics=numTopics)

    topics = []

    pprint(lda_model.print_topics(numTopics))

    for topic in lda_model.show_topics(num_topics=numTopics, formatted=False):

        topic_dict = {'words':[],'probs':[]}

        for word, prob in topic[1]:
            topic_dict['words'].append(word)
            topic_dict['probs'].append(prob)

        topics.append(topic_dict)

    return topics




if __name__ == '__main__':
    file=open_file(FILE_PATH)
    texts=get_arrs_pars(file)
    # print(texts)
    num=10
    topics= extract_topics(texts,num)
    print(topics)