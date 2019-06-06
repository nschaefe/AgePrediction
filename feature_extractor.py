import numpy as np
import re
import string
import operator


def get_words(text):
    res = re.sub('['+string.punctuation+']', '', text).split()
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


feature_file = "../data/smoking"
text_file = open(feature_file, "r")
data = text_file.read().split('\n')
symbol_length = 2
# extract word sequence up to length "symbol_length" and build histogram

hist = {}
for entry in data:
    if entry != "null":
        bag = get_bag_of_symbols(symbol_length, entry)
        for symbol in bag:
            tup = tuple(symbol)
            if tup not in hist:
                hist[tup] = 0
            hist[tup] = hist[tup]+1

sorted_hist = sorted(hist.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_hist[:10])
