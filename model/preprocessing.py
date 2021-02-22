"""
Text Clustering: preprocessing

Author: Jinraj K R <jinrajkr@gmail.com>
Created Date: 1-Apr-2020
Modified Date: 1-May-2020
===================================

``perform`` is the key performer of the application
It takes each utterance and performs the action asked to perform

"""

import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import logging
from config import config

cachedStopWords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
log = logging.getLogger("custom")
isu = config['ignore_short_utters']

def replace_synonyms(sents, syns):
    if len(syns) >= 1:
        for n, d in syns.items():
            for k, v in d.items():
                p = '(' + k.lower() + ')'
                if len(v) >= 1:
                    vl = [str(x).lower() for x in v]
                    p = '(' + k.lower() + '|'.join(vl) + ')'
                ptr = re.compile(r'\b' + p + r'\b')
                sents = [ptr.sub(n.lower(), u) for u in sents]
    return sents

def perform(action, sents, params):
    switcher = {
        "lowercase": lambda:[x.strip().lower() for x in sents],
        "remove_url": lambda: quick_replace(r'(https?ftp)\S+', sents),
        "remove_email": lambda: [' '.join([i for i in s.split() if '@' not in i]) for s in sents],
        "remove_weak_sents": lambda: [i for i in range(len(sents)) if len(sents[i].split(' ')) >= isu],
        "remove_duplicates": lambda: remove_duplicates(sents),
        "alphanumeric": lambda: [re.sub(r'([^a-zA-Z0-9\s]+?)', '', s) for s in sents],
        "extract_only_text": lambda: [re.sub(r'([^a-zA-Z\s]+?)', '', s) for s in sents],
        "alphanumeric_withspace": lambda: [' '.join(re.findall("[a-z0-9]+", s)) for s in sents],
        "extract_only_text_withspace": lambda: [' '.join(re.findall("[a-z]+", s)) for s in sents],
        "remove_stopwords": lambda: [' '.join([word for word in s.split() if word not in cachedStopWords]) for s in sents],
        "replace_by_synonyms": lambda: replace_synonyms(sents, params),
        "remove_unimportant_words": lambda: [' '.join([w for w in s.split() if w not in params]) for s in sents],
        "lemmatize": lambda: [' '.join([lemmatizer.lemmatize(word, "v") for word in s.split(" ")]) for s in sents],
        "stem": lambda: [' '.join([stemmer.stem(w) for w in s.split(" ")]) for s in sents],
        "remove_space": lambda: [' '.join(s.split()) for s in sents],
        "remove_blank_sents": lambda: [i for i in range(len(sents)) if sents[i]]
    }
    return switcher.get(action, lambda: "invalid action")()

def quick_replace(regex, sents):
    w_pttrn = re.compile(regex)
    return [w_pttrn.sub('', s) for s in sents]

def remove_duplicates(sents):
    seen = set()
    result = []
    for i in range(len(sents)):
        if sents[i] not in seen:
            seen.add(sents[i])
            result.append(i)
    return result
