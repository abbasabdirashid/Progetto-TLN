import nltk
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import spacy
from prettytable import PrettyTable

nlp = spacy.load('en_core_web_lg')

PATH_NASARI = 'utils/dd-small-nasari-15.txt'
PATH_TEXT = 'utils/muhammadali.txt'
PATH_NASARI_FULL = 'utils/dd-nasari.txt'

NONE  = 2


def get_sentences(path):
    with open(path) as file:
        lines = file.readlines()
    sentences = []
    for line in lines:
        for sentence in line.split('.'):
            if len(sentence) > 1 and not sentence.startswith('#'):
                sentences.append(sentence)
    return sentences


def split_text(sentences, size_window):
    windows = []
    index = 0
    while index < len(sentences):
        window = []
        for i in range(size_window):
            current_index = index + i
            if current_index >= len(sentences):
                break
            else:

                window.append(sentences[current_index])
        index += size_window
        windows.append(window)

    return windows


def load_nasari(path):

    with open(path, 'r') as file:
        lines = file.readlines()
    result, nasari = [], {}

    for line in lines:
        result.append(line.strip().split(';'))

    for row in result:
        b_id = row[0]
        word = row[1].lower()
        synsets = row[2:]
        nasari[word] = []
        tmp = {}

        for syn in synsets:
            syn_splitted = syn.strip().split('_')
            if len(syn_splitted) > 1:
                tmp[syn_splitted[0]] = float(syn_splitted[1])
        nasari[word].append({'b_id': b_id, 'synsets': tmp})
    return nasari


def get_nasari_vect(words, nasari):
    vectors = []
    for word in words:
        if word in nasari.keys():
            vectors.append(nasari[word][0]['synsets'])
    return vectors


def get_similarity_wo(v1, v2):
    similarity = []
    for word1 in v1:
        for word2 in v2:
            similarity.append(weighted_overlap(word1, word2))

    if len(similarity) > 0:
        return sum(similarity)/(len(v1)*len(v2)+2)*10
    else:
        return 0


def weighted_overlap(w1, w2):
    O = set(w1.keys()).intersection(w2.keys())
    # O is the set of the overlap dimension between 2 vector
    rank_acc, den = 0, 0
    if len(O):
        for i, q in enumerate(O):
            den += 1. / (2 * (i + 1))
            # ( rank of q in_w1 + rank of q in_w2 ) ^ (-1)
            rank_acc += 1. / (rank(q, [(v, k) for k, v in w1.items()]) + rank(q, [(v, k) for k, v in w2.items()]))
        return np.sqrt(pow(rank_acc, -1) / den)
    else:
        return 0.0


# get rank (len(v1) - index of q in v1)
def rank(q, v1):
    for index, item in enumerate(v1):
        if list(item)[1] == q:
            return index + 1


def tokenize(windows):
    token_of_windows = []
    stop_words = stopwords.words('english')

    for window in windows:
        token_of_window = []
        for sentence in window:
            tokens = nltk.word_tokenize(sentence.lower())
            for t in tokens:
                if len(remove_punctuation(t.lower())) > 1:
                    if remove_punctuation(t.lower()) not in stop_words:
                        token_of_window.append(t.lower())
        token_of_windows.append(token_of_window)

    return token_of_windows


def remove_punctuation(string):
    chars = '.,:;!?()”“…-'
    for c in chars:
        string = string.replace(c, '')
    string = string.replace("'s", '')
    string = string.replace("'m", '')
    string = string.replace("ai", '')
    string = string.replace("\'\'", '')
    string = string.replace("``", '')
    return string

def evaluate_similarity(token_of_sentences, nasari):
    similarities = []

    current_window = get_nasari_vect(token_of_sentences[0], nasari)

    follow_window = get_nasari_vect(token_of_sentences[1], nasari)

    similarities.append(get_similarity_wo(current_window, follow_window)/2)


    for index in range(1, len(token_of_sentences) - 1):
        prev_window = get_nasari_vect(token_of_sentences[index - 1], nasari)

        current_window = get_nasari_vect(token_of_sentences[index], nasari)

        follow_window = get_nasari_vect(token_of_sentences[index + 1], nasari)


        sim_prev = get_similarity_wo(current_window, prev_window)

        sim_follow = get_similarity_wo(current_window, follow_window)

        #print((sim_prev + sim_follow) / 2)
        similarities.append((sim_prev + sim_follow) / 2)

    prev_window = get_nasari_vect(token_of_sentences[len(token_of_sentences) - 2], nasari)

    current_window = get_nasari_vect(token_of_sentences[len(token_of_sentences) - 1], nasari)

    similarities.append(get_similarity_wo(current_window, prev_window)/2)


    return similarities


def get_best_min(similarities, index_windows_low_rank, current_windows):
    best_min = None
    max_val = 0

    for index in index_windows_low_rank:
        #If is the best
        if index > 0 and index + 1 < len(similarities):
            if similarities[index - 1] != -1 and similarities[index + 1] != -1:
                if similarities[index - 1] + similarities[index + 1] > max_val and index not in current_windows:
                    max_val = similarities[index - 1] + similarities[index + 1]
                    best_min = index

    return best_min


def find_break_points(similarities, N):
    split_points, index_windows_low_rank = [], []
    mean = sum(similarities) / len(similarities)
    print("means: {}".format(str(mean)))

    for index, sim in enumerate(similarities):
        if sim < mean and (index != 0 and index != len(similarities) - 1):
            # Perche similarities[i] = rank_windows(i+1)
            index_windows_low_rank.append(index)

    j = 0
    while j < N:
        index_min = get_best_min(np.array(similarities), index_windows_low_rank, split_points)
        if index_min is None:
            break
        #similarities[index_min] = -1
        split_points.append(index_min)
        j += 1

    return split_points


def plot_result(break_point, break_point_target, similarities, size_windows):
    plt.ylabel("Coesion intra group")
    plt.xlabel("Windows number ( windows size = {} )".format(str(size_windows)))
    plt.plot(similarities, color = "k", linewidth=1)

    for split in break_point_target:
        plt.axvline(x=split, ls="-", color='y', linewidth=5)

    for split in break_point:
        plt.axvline(x=split, ls="-", color='r', linewidth=0.5)

    plt.axvline(x=break_point[0], ls = "-", color='r', linewidth=1, label = "Breakpoints obtained")
    plt.axvline(x=break_point_target[0], ls = "-", color='y', linewidth=1, label = "Breakpoints target")

    plt.legend()
    plt.show()


def is_relevant(token):
    if token.is_stop or token.is_punct or token.is_space or token.is_digit:
        return False
    else:
        return True
    #elif token.ent_type_ == 'PERSON' || token.tag_ in ['NNP','NOUN','VERB','']


def get_occurences_of_word(sentences, word):
    occurences = 0
    for sentence in sentences:
        if word in sentence:
            occurences += 1
    return occurences


def get_cohesion_matrix(sentences):
    sentences_words, relevant_words = [], set()
    matrix = PrettyTable()

    for sentence in sentences:
        sentence_word = []
        tokens = nlp(sentence)
        for token in tokens:
            if is_relevant(token):
                sentence_word.append(token.text)
                relevant_words.add((token.text, get_occurences_of_word(sentences, token.text)))
        sentences_words.append(sentence_word)

    indexes = [index for index in range(len(sentences)+1)]
    indexes[0] = 'SENTENCES'
    matrix.field_names = indexes

    for w in relevant_words:
        if w[1] > 3:
            row = [w[0]]
            for index, sentence in enumerate(sentences):
                if w[0] in sentence:
                    row.append(1)
                else:
                    row.append('')
            matrix.add_row(row)

    return matrix
if __name__ == '__main__':
    # 1 Get sentences
    sentences = get_sentences(PATH_TEXT)
    nasari = load_nasari(PATH_NASARI_FULL)

    break_point_sentence_target = [5, 17, 36, 48, 51]
    size_windows = 5
    for index, val in enumerate(break_point_sentence_target):
        break_point_sentence_target[index] = val/size_windows

    # 2 Windowing
    windows = split_text(sentences, size_windows)
    print("Number of windows: {} \n Size of windows: {}".format(str(len(windows)), size_windows))

    # 3 Tokenizing
    token_of_sentences = tokenize(windows)
    # 4 Evaluate similairty between windowsx
    similarities = evaluate_similarity(token_of_sentences, nasari)
    # 5 Find break point
    break_point = find_break_points(similarities, len(break_point_sentence_target))
    # # 6 Get result
    plot_result(break_point, break_point_sentence_target, similarities, size_windows)
    #
    matrix = get_cohesion_matrix(sentences)
    matrix.vrules = NONE
    #
    # print(matrix)