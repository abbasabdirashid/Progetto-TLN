from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

stop_words_set = set()
lemmatizer = WordNetLemmatizer()


def build_words_path_set(path):
    res = set()
    with open(path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        res.add(str(line.rstrip()))
    return res


def filter_stopword_from_sentence(sentence):
    relevant_words = set()
    for s in sentence.split(' '):
        if s not in stop_words_set:
            relevant_words.add(remove_punctuation(s))
    return relevant_words


def get_words_from_examples(examples):
    words = set()
    if len(examples) > 0:
        for example in examples:
            for word in example.split():
                words.add(remove_punctuation(word))
    return words


def get_words_from_definition(param):
    words = set()
    for word in param.split(' '):
        if words not in stop_words_set:
            words.add(word)
    return words


def remove_punctuation(string):
    chars = '.,:;!?()”“…-'
    for c in chars:
        string = string.replace(c, '')
    string = string.replace("’s", '')
    string = string.replace("’s", '')
    return string


def intersection(first_set, second_set):
    intersection_set = set()
    for var in first_set:
        if var in second_set:
            intersection_set.add(var)
    return intersection_set


def union(s1, s2):
    res = set()
    for var in s1:
        res.add(var)
    for var in s2:
        if var not in res:
            res.add(var)
    return res


def get_wordnet_ctx(sense):
    res = set()
    words_examples = get_words_from_examples(sense.examples())
    words_definition = get_words_from_definition(sense.definition())
    contex = union(words_examples, words_definition)

    # Hyponym contex
    for hyponym in sense.hyponyms():
        hyponym_ctx_word = union(get_words_from_examples(hyponym.examples()),
                                 get_words_from_definition(hyponym.definition()))
        contex = union(hyponym_ctx_word, contex)

    # hypernym contex
    for hypernym in sense.hypernyms():
        hypernym_ctx_word = union(get_words_from_examples(hypernym.examples()),
                                  get_words_from_definition(hypernym.definition()))
        contex = union(hypernym_ctx_word, contex)

    # Filter stopword
    for w in contex:
        if w not in stop_words_set:
            res.add(w)

    return res


def lesk_algorithm(word, sentence):
    senses = wn.synsets(word.strip())
    if len(senses) == 0:
        return None
    else:
        best_sense = wn.synsets(word)[0]
        max_overlap = set()
        context = filter_stopword_from_sentence(sentence)
        for sense in wn.synsets(word):
            signature = get_wordnet_ctx(sense)
            overlap = intersection(signature, context)
            if len(overlap) > len(max_overlap):
                max_overlap = overlap
                best_sense = sense
        return best_sense