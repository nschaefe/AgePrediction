import string
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))


def get_words(text):
    res = text.translate(translator).split()
    #res = re.sub('['+string.punctuation+']', ' ', text).split()
    for i in range(0, len(res)):
        res[i] = res[i].strip()
    return res
