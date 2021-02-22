
import os
application_name = "utterance_clustering"
application_path = os.getcwd()
print("application path: {}".format(application_path))

config = {
    "application_path": application_path,
    "predefined_vectors_path": application_path + "/predefined_vectors",
    "min_input_size": 10,
    "ignore_short_utters": 3,
    "bert_path": application_path + "/predefined_vectors/bert_files",
    "float_decimal_limit": 1,
    "auto_generate_synonyms_modes": {
        "strict": [1, 5, 4],
        "moderate": [1, 4, 4],
        "loose": [1, 3, 3]
    },
    "min_word_length": 3,
    "min_synonyms_per_value": 2,
    "max_synonyms_per_value": 8,
    "preprocess_for_slots":[
        "lowercase",
        "remove_email",
        "remove_url",
        "remove_stopwords",
        "remove_unimportant_words",
        "lemmatize",
        "remove_space",
        "remove_weak_sents",
        "output_format"
    ],
    "preprocess_for_intents": [
        "lowercase",
        "remove_email",
        "remove_url",
        "remove_space",
        "remove_weak_sents",
        "output_format",
        "remove_stopwords",
        "remove_unimportant_words",
        "lemmatize",
        "replace_by_synonyms",
        "remove_space",
        "remove_blank_sents"
    ],
    "output_format": [
        "alphanumeric",
        "extract_only_text"
    ],
    "execution": [
        "preprocess",
        "slots",
        "embed",
        "cluster",
        "intents"
    ],
    "synonyms_generating_types": ["auto", "custom"],
    "embedding_models": ["tfidf", "word2vec", "w2v_tfidf", "bert"],
    "clustering_models": ["custom", "ahc", "kmeans", "hdbscan"]

}



