"""
Text Clustering: Identify_Slots

Author: Jinraj K R <jinrajkr@gmail.com>
Created Date: 1-Apr-2020
Modified Date: 1-May-2020
===================================

This class constructor takes array of utterances and four integer parameter
which actually defines the mode of generating synonyms (auto_generate_synonym_modes - explained in index.py file).
> performs cleaning the utterances like given in the global variable ``steps``.
> identify synonyms and replace the words in the utterances by synonym name
> generates maximum number of clusters
and returns clustered utterances

``execute`` is the main method which initiates the execution
"""

import itertools
import time as t
import numpy as np
import pandas as pd
from config import config

class Identify_Slots:
    def __init__(self, sentences, slots_config):
        self.sentences = sentences
        self.sentence_token = [s.split() for s in sentences]
        self.distance = slots_config[0]
        self.min_occ_of_neighbour_keys = slots_config[1]
        self.min_values_in_slot = slots_config[2]

    def get_lcr(self, sents):
        df = pd.DataFrame()
        d = self.distance
        for sent in sents:
            for i in range(d, len(sent)-d, 1):
                ar = [sent[l] for l in range(i - d, i + d + 1)]
                df = df.append(pd.DataFrame([ar]), ignore_index=True)
        return df

    def possible_slots(self):
        ln = len(self.sentence_token)
        t0 = t.time()
        df = self.get_lcr(self.sentence_token)
        print("dataframe rows - {}, created in {:.3f} s".format(df[0].count(), t.time()-t0))

        print("filtering dataframe started")
        t1 = t.time()
        selected_words_grp = []
        rs = df.groupby(df.columns.tolist(), as_index=False).size()
        nrs = rs[rs['size'] >= self.min_occ_of_neighbour_keys]

        if len(nrs):
            cnrs = nrs[nrs.groupby([0,2])[1].transform("count") >= self.min_values_in_slot]
            if len(cnrs):
                fcnrs = cnrs.groupby([0,2]).agg(list)[1].tolist()
                selected_words_grp.extend(fcnrs)

            rnrs = nrs[nrs.groupby([0, 1])[2].transform("count") >= self.min_values_in_slot]
            if len(rnrs):
                frnrs = rnrs.groupby([0, 1]).agg(list)[2].tolist()
                selected_words_grp.extend(frnrs)

            lnrs = nrs[nrs.groupby([1, 2])[0].transform("count") >= self.min_values_in_slot]
            if len(lnrs):
                flnrs = lnrs.groupby([1, 2]).agg(list)[0].tolist()
                selected_words_grp.extend(flnrs)
        print("filtering dataframe finished in {:.3f} s".format(t.time() - t1))

        cleaned_words_grp = self.clean_words(selected_words_grp)

        t3 = t.time()
        print("finding synonyms of slot values")
        final_slots = self.get_syn_of_slots(cleaned_words_grp)
        print("finding synonyms of slot values finished in {:.3f} s".format(t.time() - t3))
        return final_slots

    def get_syn_of_slots(self, filtered_slots):
        allwords = []
        for ar in self.sentence_token:
            allwords = allwords + ar
        allwords = list(set(allwords))

        from model.identify_synonyms import get_syn
        fnl = []
        for arr in filtered_slots:
            slt = {}
            for w in arr:
                syns = get_syn(w)
                isyns = list(np.intersect1d(syns, allwords))
                if w in isyns:
                    isyns.remove(w)
                vls = []
                for kv in fnl:
                    kk = list(kv.keys())
                    vv = list(kv.values())
                    vv.append(kk)
                    vls.extend(vv)
                vls = list(itertools.chain(*vls))
                intrs = list(np.setdiff1d(isyns, vls))
                if config["max_synonyms_per_value"] >= len(intrs) >= config["min_synonyms_per_value"]:
                    slt[w] = intrs
            if len(slt.keys()) >= self.min_values_in_slot:
                fnl.append(slt)
        return fnl

    def slots_by_emb_model(self, grouped_words):
        from model.features import Features
        from model.cluster_custom import CustomModel
        words = []
        for ar in grouped_words:
            words = words + ar
        words = list(set(words))

        vec = Features(words).vec_bert()
        slots = CustomModel(words).cluster(vec, self.min_values_in_slot, 1.0)
        return [s for s in slots.values()]

    def clean_words(self, lst):
        lst = remove_short_words(lst)
        lst = check_min_values(lst, self.min_values_in_slot)
        lst = remove_dup_list(lst)
        fnl = []
        for v in lst:
            if len(fnl) >= 1:
                afnl = itertools.chain(*fnl)
                temp = list(np.setdiff1d(v, afnl))
                if len(temp) >= self.min_values_in_slot:
                    fnl.append(temp)
            else:
                fnl.append(v)
        return fnl


def remove_dup_dict(a_of_d):
    dict_cnts = []
    for d in a_of_d:
        l = dict_cnts[len(dict_cnts)-1] if len(dict_cnts) > 0 else 0
        dict_cnts.append(len(d.keys()) + l)

    dct = {}
    for dc in a_of_d:
        for k, v in dc.items():
            dct[k] = v

    temp = {}
    for k, v in dct.items():
        if k not in str(temp):
            vs = []
            for w in v:
                if w not in str(temp):
                    vs.append(w)
            temp[k] = vs

    fnl = []
    ct = 0
    tp = {}
    for k, v in temp.items():
        if ct in dict_cnts:
            fnl.append(tp)
            tp = {}
        else:
            tp[k] = v
        ct = ct + 1
    return fnl


def remove_dup_list(l_of_l):
    lst = []
    for k in l_of_l:
        if k not in lst:
            lst.append(k)
    return lst


def remove_short_words(l_of_l):
    return [[w for w in ar if len(w) >= config["min_word_length"]] for ar in l_of_l]


def check_min_values(l_of_l, min_vals):
    return [i for i in l_of_l if len(i) >= min_vals]


























