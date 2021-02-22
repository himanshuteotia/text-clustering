"""
Text Clustering: validation

Author: Jinraj K R <jinrajkr@gmail.com>
Created Date: 1-Apr-2020
Modified Date: 1-May-2020
===================================

This function takes input paramters and validates
Returns a alert message if there is any invalid parameter
If the validation gets success, it creates an instance of BotClusters and returns the object

"""

from index import BotClusters
import json
from config import config
import uuid

synonyms_generating_types = config["synonyms_generating_types"]
auto_generate_synonyms_modes = config["auto_generate_synonyms_modes"]
output_format = config["output_format"]
clustering_models = config["clustering_models"]
embedding_models = config["embedding_models"]

def uniqueid():
    pid = uuid.uuid4().hex
    print("generated new process id - {}".format(pid))
    return pid

def validate(params):
    process_id = uniqueid()
    botname = "test_bot"
    excel_data = []
    synonyms_generating_type = synonyms_generating_types[0]
    custom_synonyms = {}
    auto_generate_synonyms_mode = "moderate"
    remove_unimportant_words = []
    output_utterances_type = "extract_only_text"
    min_samples = 3
    clustering_model = clustering_models[0]
    embedding_model = embedding_models[3]
    no_of_clusters = 5
    merge_similar_clusters = 90
    ignore_similarity = 30

    if "botname" in params:
        if (params["botname"]).strip()=="" or " " in (params["botname"]).strip():
            return "botname is alphanumeric with underscores. Must not contain any space. ex- pizza_bot_01"
        else:
            botname = params["botname"]
    else:
        return "botname is required"

    if "excel_data" in params:
        if len(params["excel_data"]) >= 10:
            excel_data = params["excel_data"]
        else:
            return "excel_data must contain atleast 10 utterances"
    else:
        return "excel_data is required"

    if "adv_settings" in params:
        if "synonyms_generating_type" in params["adv_settings"]:
            if params["adv_settings"]["synonyms_generating_type"] in synonyms_generating_types:
                synonyms_generating_type = params["adv_settings"]["synonyms_generating_type"]
                if synonyms_generating_type == synonyms_generating_types[1]:
                    if synonyms_generating_types[1] in params["adv_settings"]:
                        if type(params["adv_settings"][synonyms_generating_types[1]]) is dict:
                            if bool(params["adv_settings"][synonyms_generating_types[1]]):
                                custom_synonyms = params["adv_settings"][synonyms_generating_types[1]]
                            else:
                                return "custom_synonyms cannot be empty"
                        else:
                            return "custom_synonyms must be dictionary"
                    else:
                        return "custom_synonyms is required of synonyms_generating_type = custom_synonyms"
            else:
                return "synonyms_generating_type is not valid"

        if "auto_generate_synonyms_mode" in params["adv_settings"]:
            if params["adv_settings"]["auto_generate_synonyms_mode"] in auto_generate_synonym_modes.keys():
                auto_generate_synonyms_mode = params["adv_settings"]["auto_generate_synonyms_mode"]
            else:
                return "auto_generate_synonyms_mode is not valid"

        if "clustering_model" in params["adv_settings"]:
            if params["adv_settings"]["clustering_model"] in clustering_models:
                clustering_model = params["adv_settings"]["clustering_model"]
            else:
                return "clustering_model is not valid"

        if "embedding_model" in params["adv_settings"]:
            if params["adv_settings"]["embedding_model"] in embedding_models:
                embedding_model = params["adv_settings"]["embedding_model"]
            else:
                return "embedding_model is not valid"

        if "remove_unimportant_words" in params["adv_settings"]:
            if isinstance(params["adv_settings"]["remove_unimportant_words"], list):
                remove_unimportant_words = params["adv_settings"]["remove_unimportant_words"]
            else:
                return "remove_unimportant_words must be an array"

        if "min_samples" in params["adv_settings"]:
            if isinstance(params["adv_settings"]["min_samples"], int):
                if params["adv_settings"]["min_samples"] <= 1:
                    return "min_samples must be >= 2"
                else:
                    min_samples = params["adv_settings"]["min_samples"]
            else:
                return "min_samples must be an integer"

        if "no_of_clusters" in params["adv_settings"]:
            if isinstance(params["adv_settings"]["no_of_clusters"], int):
                if params["adv_settings"]["no_of_clusters"] <= 1:
                    return "no_of_clusters must be >= 2"
                else:
                    no_of_clusters = params["adv_settings"]["no_of_clusters"]
            else:
                return "min_samples must be an integer"

        if "merge_similar_clusters" in params["adv_settings"]:
            if isinstance(params["adv_settings"]["merge_similar_clusters"], int):
                if 20 <= params["adv_settings"]["merge_similar_clusters"]:
                    merge_similar_clusters = round(params["adv_settings"]["merge_similar_clusters"], 1)
                else:
                    return "merge_similar_clusters must be < 100 and >= 20"
            else:
                return "merge_similar_clusters must be an integer"

        if "ignore_similarity" in params["adv_settings"]:
            if isinstance(params["adv_settings"]["ignore_similarity"], int):
                if params["adv_settings"]["ignore_similarity"] < merge_similar_clusters:
                    ignore_similarity = params["adv_settings"]["ignore_similarity"]
                else:
                    return "ignore_similarity must be < merge similar clusters, try to provide the lowest like 10 or 20 etc..."
            else:
                return "ignore_similarity must be an integer"

    return BotClusters(dict(
        process_id=process_id,
        botname=botname,
        excel_data=excel_data,
        synonyms_generating_type=synonyms_generating_type,
        custom_synonyms=custom_synonyms,
        auto_generate_synonyms_mode=auto_generate_synonyms_mode,
        remove_unimportant_words=remove_unimportant_words,
        output_utterances_type=output_utterances_type,
        min_samples=min_samples,
        clustering_model=clustering_model,
        embedding_model=embedding_model,
        no_of_clusters=no_of_clusters,
        merge_similar_clusters=merge_similar_clusters,
        ignore_similarity=ignore_similarity
    ))
