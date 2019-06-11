import numpy as np
import re
import string
import operator

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

def get_words(text):
    res = text.translate(translator).split()
    #res = re.sub('['+string.punctuation+']', ' ', text).split()
    for i in range(0, len(res)):
        res[i] = res[i].strip()
    return res


def get_bag_of_symbols(max_len, string):
    words = get_words(string)
    bag = []
    window_size = 1
    while window_size <= max_len:
        for s_i in range(0, len(words)-window_size+1):
            symbol = []
            for win_i in range(s_i, s_i+window_size):
                symbol.append(words[win_i])
            bag.append(symbol)
        window_size = window_size+1
    return bag


def get_word_hist(data, symbol_length):
    hist = {}
    for entry in data:
        if entry != "null":
            bag = get_bag_of_symbols(symbol_length, entry)
            for symbol in bag:
                tup = tuple(symbol)
                if tup not in hist:
                    hist[tup] = 0
                hist[tup] = hist[tup]+1

    sorted_hist = sorted(
        hist.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_hist[:10])
    return sorted_hist


def relable(data, keywords):
    for i in range(0, len(data)):
        label = data[i]
        clean_label = ' '.join(get_words(label))

        labeled = False
        for keyword in keywords:
            if keyword in clean_label:
                data[i] = keyword
                labeled = True
                break
        if not labeled:
            data[i] = 'null'


feature_file = "../data/smoking"
text_file = open(feature_file, "r")
data = text_file.read().split('\n')
symbol_length = 2

hist = get_word_hist(data, symbol_length)

# example for relabeling
# keywords = ['nefajcim', 'fajcim pravidelne', 'fajcim prilezitostne', 'fajcim']
# relable(data, keywords)
# print(data[:10])
