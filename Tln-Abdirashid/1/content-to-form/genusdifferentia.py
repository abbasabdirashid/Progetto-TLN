from io import open
import csv
import re

from collections import Counter
import spacy
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from spacy_wordnet.wordnet_annotator import WordnetAnnotator


lemmatizer = WordNetLemmatizer()
nlp = spacy.load(r"/opt/miniconda3/lib/python3.8/site-packages/en_core_web_sm/en_core_web_sm-3.2.0")
nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
PUNTUATION_SET = '.,:;!?()”“…'

DEFINITIONS= "utils/tlndicaro_1.1_defs - Foglio1.csv"
STOP_WORDS="utils/stop_words_FULL.txt"


def load_def(path):
    result=[]
    with open(DEFINITIONS, newline='') as csvfile:
        r = csv.reader(csvfile)
        for i,row in enumerate(r):
            #result.append(ele)
            #x=' '.join(row)
            if i!=0:
                result.append(
                    {'Partecipante': row[0],
                     'Courage':row[1],
                     'Paper':row[2],
                     'Apprehension':row[3],
                     'Sharpener':row[4]}
                )
    return result


def load_stopwords(STOP_WORDS):
    result=[]
    with open(STOP_WORDS,'r') as f:
        lines=f.readlines()
    for line in lines:
        line=line.strip('\n')
        result.append(line)

    return result


def remove_punctuation(string):
    chars = '.,:;!?()”“…-'
    for c in chars:
        string = string.replace(c, '')
    string = string.replace("’s", '')
    return string



def get_dict_items(list,string):
    li=[]
    if string=='Courage':
        for i in list:
            for l in i.keys():
                if l=='Courage' and i.get(l):
                    li.append(i.get(l))

    if string=='Paper':
        for i in list:
            for l in i.keys():
                if l=='Paper' and i.get(l):
                    li.append(i.get(l))

    if string=='Apprehension':
        for i in list:
            for l in i.keys():
                if l=='Apprehension' and i.get(l):
                    li.append(i.get(l))
    if string=='Sharpener':
        for i in list:
            for l in i.keys():
                if l=='Sharpener' and i.get(l):
                    li.append(i.get(l))


    return li


def get_syn_score(synset,context):
    stop_words = stopwords.words('english')
    definition = nlp(remove_punctuation(synset.definition()))
    contex_nlp = nlp(context)
    definition_tokenized = nlp(' '.join([lemmatizer.lemmatize(str(t)) for t in definition if str(t) not in stop_words]))
    context_tokenized = nlp(' '.join([str(t) for t in contex_nlp if str(t) not in stop_words]))
    return definition_tokenized.similarity(context_tokenized)


def get_best_synset(synsets,context):
    best_synset, best_score = None, 0

    for s in synsets:
        score = get_syn_score(s, context)
        #print('Hyper visited {} with score {}'.format(hyp, score))
        if score > best_score and s._pos == 'n':
            best_synset = s
            best_score = score
        for hyp in s.hyponyms():
            score = get_syn_score(hyp, context)
            #print('Hyp visited {} with score {}'.format(hyp, score))
            if score > best_score and s._pos == 'n':
                best_synset = hyp
                best_score = score

    return best_synset, best_score


def content_to_form(definition_list, max_terms_in_context, max_genus):
    domain, context = [], []
    stop_words = stopwords.words('english')

    for definition in definition_list:
        # 1 Estrazione termini rilevanti
        text = nlp(definition)
        subjs = filter(lambda token: token.dep_ == 'ROOT', text)

        relevant_words = filter(lambda token:
                                token.text.lower() not in stop_words and
                                token.text.lower() not in PUNTUATION_SET, text)
        relevant_words = list(relevant_words)
        # print(relevant_words)

        # 2 Costruzione Domain + Context
        # Domain = soggetti delle frasi + wordnet domain of them
        # Context = sfera semantica i termini rilevanti di ogni definizione
        domain.extend(list(map(lambda t: t.text, subjs)))
        for token in relevant_words:
            domain.extend(token._.wordnet.wordnet_domains())
            context.append(token.text)

    # 3 Termini con frequenza maggiore
    candidate_genus = Counter(domain).most_common(max_genus)
    common_context = Counter(context).most_common(max_terms_in_context)
    s_context = ' '.join(list(map(lambda c: lemmatizer.lemmatize(c[0]), common_context)))

    print('Context obtained : {}'.format(s_context))
    print('Genus obtained : {}'.format(candidate_genus))

    # 4 Per ogni synset di un dominio, cerco tra gli iponimi quello che  ha score
    # di similarità maggiore rispetto al contesto
    best_synset = None
    best_score = 0
    for lemma in candidate_genus:
        # 5 Valutazione di ogni synset esplorato
        synset, score = get_best_synset(wn.synsets(lemmatizer.lemmatize(lemma[0])), s_context)
        if score > best_score:
            best_synset = synset
            best_score = score

    return best_synset, best_score






def main():
    data=load_def(DEFINITIONS)
    #print(data)
    target=['Courage','Paper','Apprehension','Sharpener']
    max_terms_in_context=20
    max_genus=15
    for i, def_list in enumerate(data):


        if i<len(target):
            print('*'*100)
            print('Target terms : {}'.format(target[i]))
            list_def=get_dict_items(data,target[i])
            #print(list_def)
            syns, sim = content_to_form(list_def, max_terms_in_context, max_genus)
            #print(syns,sim)
            print(f'Similarity: {sim} with {syns} \nDef : {syns.definition()}\n')
            print('*'*100)
            print('\n')



if __name__=='__main__':
    main()

