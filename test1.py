from file_mgmt import read_excel
import time as t
from model.handler import _main
data_sample = [
    'a weeker person cannot become a king because he can be killed anytime',
    'reset my @ % & password',
    'reset my machine _ !!',
    'king can be killed anytime so we need make him strong',
    'our prince is as strong as rock',
    'your prince was as weak as feather',
    'queen is always a young girl',
    'prince is a boy will be king',
    'princess is a girl will be queen',
    'queen is looking like a godess',
    'queen was once a young girl',
    'reset my passcode',
    'a smart lady can become queen in any kingdom',
    'reset my win password',
    "She doesn’t study German on Monday.",
    "Does she live in Paris?",
    "He doesn’t teach math.",
    "Cats hate water.",
    "Every child likes an ice cream.",
    "My brother takes out the trash.",
    "The course starts next Sunday.",
    "She swims every morning.",
    "I don’t wash the dishes.",
    "We see them every week.",
    "I don’t like tea.",
    "When does the train usually leave?",
    "She always forgets her purse.",
    "You don’t have children.",
    "I and my sister don’t see each other anymore.",
    "They don’t go to school tomorrow.",
    "He loves to play basketball.",
    "He goes to school.",
    "The Earth is spherical",
    "Julie talks very fast.",
    "My brother’s dog barks a lot."]


def input_data(file_path):
    if file_path != "":
        return read_excel(file_path)
    else:
        return data_sample

# input parameters
input_params = {
    "botname": "pizza_bot",
    "excel_data": input_data("C:/Users/Jinraj/Desktop/MLProjects/Utters.xlsx"),
    "adv_settings": {
        "synonyms_generating_type": "auto",  # "auto" OR "custom"
        "custom_synonyms": {},
        "auto_generate_synonyms_modes": "loose",  # loose, moderate or strict
        "remove_unimportant_words": [],
        "output_utterances_type": "extract_only_text",  # extract_only_text or alphanumeric
        "min_samples": 2,
        "embedding_model": "w2v_tfidf",
        "clustering_model": "custom",
        "no_of_clusters": 10,
        "merge_similar_clusters": 90,
        "ignore_similarity": 30
    }
}

t1 = t.time()
res = _main(input_params, "intents")
print(res)
print("----- {} seconds -----".format(t.time() - t1))