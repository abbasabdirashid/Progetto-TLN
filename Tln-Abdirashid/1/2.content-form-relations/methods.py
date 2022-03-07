import string

import nltk
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

LIMIT_hyponyms = 1041


def get_average_len_def():
    sum=0
    for s, sense in enumerate(wn.all_synsets()):
        definition= sense.definition()
        definition_tokens=word_tokenize(definition)
        sum+= len(definition_tokens)
    return sum/(s+1),s+1


def get_clear_words(line, type=None):
    stop_words = set(stopwords.words('english'))
    puncts = ['', '‘', '’', '”', '“']

    words = word_tokenize(line)
    if type == 'stem':
        filtered_sentence = [porter.stem(w.lower().strip(string.punctuation)) for w in words if
                             not w.lower().strip(string.punctuation) in stop_words and w.lower().strip(
                                 string.punctuation) not in puncts]
    elif type == 'lem':
        filtered_sentence = [lemmatizer.lemmatize(w.lower().strip(string.punctuation)) for w in words if
                             not w.lower().strip(string.punctuation) in stop_words and w.lower().strip(
                                 string.punctuation) not in puncts]
    else:
        filtered_sentence = [w.lower().strip(string.punctuation) for w in words if
                             not w.lower().strip(string.punctuation) in stop_words and w.lower().strip(
                                 string.punctuation) not in puncts]
    return filtered_sentence



def bag_of_words(contexts, definition):

    best_sense = None

    if len(contexts) == 0:
        return best_sense
    else:

        max = float('-inf')


        sentence_context = set(get_clear_words(definition,'stem'))


        for context in contexts:

            score = (len(sentence_context.intersection(list(set(get_clear_words(context['context'],'stem'))))))

            if score > max:
                max = score
                best_sense = context['synset']


    return best_sense

def get_context_of_synset(sense):

    examples = sense.examples()
    gloss = sense.definition()

    res = gloss

    for example in examples:
        res += example

    return res



def create_contexts(elem):

    context_wn = []

    try:
        synsets = wn.synsets(elem)
    except Exception:
        # print('it is not in wordnet\n\n\n')
        return '', False

    for synset in synsets:


        context_synset = get_context_of_synset(synset)


        try:
            hyponyms = synset.hyponyms()
            for sense in hyponyms:
                context_synset += get_context_of_synset(sense)
        except Exception:
            pass

        try:
            hypernyms = synset.hypernyms()
            for sense in hypernyms:
                context_synset += get_context_of_synset(sense)
        except Exception:
            pass



        # context created
        context_wn.append({'synset': synset, 'context': context_synset})



    return context_wn,True


def get_height_of_sense(best_sense):
    return min([len(path) for path in best_sense.hypernym_paths()])


def get_avg_height_of_definitions():
    avg_definitions=[]
    for s,sense in enumerate(wn.all_synsets()):
        definition= sense.definition()
        # print(definition)
        clear_definition = get_clear_words(definition, type = None)
        #print(clear_definition)
        sum_height=0
        i_h=0
        for elem in clear_definition:
            contexts_with_synsets = create_contexts(elem)
            if contexts_with_synsets[1]:

                best_sense = bag_of_words(contexts_with_synsets[0], definition)
                try:
                    sum_height += get_height_of_sense(best_sense)
                    i_h+=1
                except Exception:
                    pass



        if sum_height!=0:
            avg_definition_height = sum_height/i_h
            avg_definitions.append(avg_definition_height)

        if s == 1000: # imposto un limite per effettuare tests rapidi
            break

    try:
        return sum(avg_definitions)/len(avg_definitions)
    except Exception:
        return 'Error : division by 0'


def get_arr_rel(sense, type,arr,count):



    if type == 'hyponyms':

        if count == LIMIT_hyponyms:
            return arr
        try:
            relation = sense.hyponyms()
        except Exception:
            return arr

    elif type == 'hypernyms':

        try:
            relation = sense.hypernyms()
        except Exception:
            return arr

    elif type == 'meronyms':

        try:
            relation = sense.part_meronyms()
        except Exception:
            return arr

    elif type == 'holonymys':

        try:
            relation = sense.substance_holonyms()
        except Exception:
            return arr




    if len(relation) == 0:
        return arr

    else:

        for sense in relation:

            arr.append(sense)

            try:
                arr = get_arr_rel(sense, type, arr,count+1)
            except Exception:
                pass

        return arr


def update_relation_set(sense, definition,relational_set,type):

    relations_in_def = []

    if type != 'synonyms' and type != 'antonyms':
        # rel transitive
        rel_set = get_arr_rel(sense, type, [],0)

        for rel in rel_set:
            for lemma in rel.lemmas():
                if lemma.name() in definition:
                    relations_in_def.append(lemma.name())

        if len(relations_in_def) != 0:
            relational_set[type][sense] = relations_in_def

    else:

        if type == 'synonyms':

            for lemma in sense.lemmas():
                if lemma.name() in definition:
                    relations_in_def.append(lemma.name())

        elif type == 'antonyms':

            for lemma in sense.lemmas():
                if lemma.antonyms():
                    for antonym in lemma.antonyms():
                        if antonym.name() in definition:
                            relations_in_def.append(lemma.name())

        if len(relations_in_def) != 0:
            relational_set[type][sense] = relations_in_def

    return relational_set


def get_relational_set():

    relational_set = {'hypernyms':{},'hyponyms':{},'meronyms':{}, 'holonymys':{}, 'antonyms':{},'synonyms':{}}

    for s, sense in enumerate(wn.all_synsets()):

        definition = sense.definition()

        for type in relational_set.keys():
            relational_set = update_relation_set(sense,definition,relational_set,type)

    return relational_set