"""
Text Clustering: BotClusters

Author: Jinraj K R <jinrajkr@gmail.com>
Created Date: 1-Apr-2020
Modified Date: 1-May-2020
===================================

This class constructor takes array of utterances and other required parameters,
> performs cleaning the utterances like given in the global variable ``steps``.
> identify synonyms and replace the words in the utterances by synonym name
> generates maximum number of clusters
and returns clustered utterances

``execute`` is the main method which initiates the execution
"""

from model.clustering_models import ClusteringModels
from model.preprocessing import perform
from model.identify_slots import Identify_Slots
from model.features import Features
from config import config
import time as t
import json

auto_generate_synonyms_modes = config["auto_generate_synonyms_modes"]
slots_steps = config["preprocess_for_slots"]
intents_steps = config["preprocess_for_intents"]
output_format = config["output_format"]
synonyms_generating_types = config["synonyms_generating_types"]
clustering_models = config["clustering_models"]
embedding_models = config["embedding_models"]
execution_steps = config["execution"]


class BotClusters:
    def __init__(self, params):
        self.process_id = params["process_id"]
        self.botname = params["botname"]
        self.excel_data = params["excel_data"]
        self.synonyms_generating_type = params["synonyms_generating_type"]
        self.synonyms = params["custom_synonyms"]
        self.auto_generate_synonyms_mode = params["auto_generate_synonyms_mode"]
        self.remove_unimportant_words = params["remove_unimportant_words"]
        self.output_utterances_type = params["output_utterances_type"]
        self.min_samples = params["min_samples"]
        self.clustering_model = params["clustering_model"]
        self.embedding_model = params["embedding_model"]
        self.no_of_clusters = params["no_of_clusters"]
        self.merge_similar_clusters = params["merge_similar_clusters"]
        self.ignore_similarity = params["ignore_similarity"]

        self.return_type = None
        self.app_dict = {}
        self.app_dict["step_output"] = None
        self.app_dict["output_sentences"] = None
        self.app_dict["slots"] = None
        self.app_dict["vectors"] = None
        self.app_dict["clusterer"] = None
        self.app_dict["model"] = None
        self.app_dict["intents"] = None
        self.app_dict["twodim"] = None

        params["excel_data"] = len(params["excel_data"])
        print(json.dumps(params, indent=4))

    def identify_possible_slots(self):
        if self.synonyms_generating_type == synonyms_generating_types[0]:
            self.synonyms = Identify_Slots(self.app_dict["step_output"], auto_generate_synonyms_modes[self.auto_generate_synonyms_mode]).possible_slots()

    def run_step(self, step, utterances):
        params = ""
        if step == "replace_by_synonyms":
            params = self.synonyms if self.synonyms else "-"
        if step == "remove_unimportant_words":
            params = "_" if len(self.remove_unimportant_words) == 0 else self.remove_unimportant_words
        print('running the step - {}'.format(step))
        t1=t.time()
        if params == "-":
            res = utterances
        else:
            res = perform(step, utterances, params)
        print('finished the step - {} in {:.3f} s'.format(step, t.time() - t1))
        return res

    def loop_steps(self):
        steps = slots_steps if self.return_type == "slots" else intents_steps
        for step in steps:
            if self.app_dict["step_output"] == "" or self.app_dict["step_output"] is None:
                self.app_dict["step_output"] = self.excel_data
            step_out = self.run_step(self.output_utterances_type if step == "output_format" else step, self.app_dict["step_output"])

            if step == "remove_weak_sents":
                diff = len(self.app_dict["step_output"]) - len(step_out)
                self.app_dict["step_output"] = [self.app_dict["step_output"][i] for i in step_out]
                if diff >= 1:
                    print("removed {} weak sentences".format(diff))

            elif step=="output_format":
                distinct_inds = perform("remove_duplicates", step_out, "")
                disctint_set = [step_out[i] for i in distinct_inds]
                duplicates = len(step_out) - len(disctint_set)
                if duplicates > 0:
                    print("removed {} duplicates".format(duplicates))

                clean_inds = perform("remove_blank_sents", disctint_set, "")
                final_set = [disctint_set[i] for i in clean_inds]
                empty_set = len(disctint_set) - len(final_set)
                if empty_set>0:
                    print('removed {} weak sentences'.foramt(empty_set))

                self.app_dict["output_sentences"] = final_set
                self.app_dict["step_output"] = final_set

            elif step == "remove_blank_sents":
                self.app_dict["step_output"] = [self.app_dict["step_output"][i] for i in step_out]
                self.app_dict["output_sentences"] = [self.app_dict["output_sentences"][i] for i in step_out]

            else:
                self.app_dict["step_output"] = step_out

    def run_embed_model(self):
        ft = Features(self.app_dict["step_output"])
        if self.embedding_model == 'tfidf':
            self.app_dict["vectors"] = ft.vec_tfidf()
        elif self.embedding_model == 'word2vec':
            self.app_dict["vectors"] = ft.vec_gensim_w2v()
        elif self.embedding_model == 'w2v_tfidf':
            self.app_dict["vectors"] = ft.vec_w2c_tfidf()
        elif self.embedding_model == 'bert':
            self.app_dict["vectors"] = ft.vec_bert()
        elif self.embedding_model == 'bert_hdp':
            self.app_dict["vectors"] = ft.vec_bert_hdp()

    def run_clustering_model(self):
        cm = ClusteringModels(self.app_dict["step_output"], self.app_dict["vectors"])
        kwargs = {}
        if self.clustering_model in ['kmeans', 'ahc']:
            kwargs = {'n_clusters': self.no_of_clusters}
        elif self.clustering_model == 'custom':
            kwargs = {'merge_similar_clusters': self.merge_similar_clusters, 'ignore_similarity': self.ignore_similarity}
        md = cm.cluster(self.clustering_model, kwargs)
        self.app_dict["model"] = md
        self.app_dict["clusterer"] = cm
        cm = ""

    def check_embedding(self):
        print("running embedding model - {}".format(self.embedding_model))
        t1 = t.time()
        self.run_embed_model()
        print("finished embedding using {} in - {:.3f} s. i.e., {} sentences/sec".format(
            self.embedding_model, t.time() - t1, round(len(self.app_dict["step_output"])/(t.time() - t1))))

    def check_clustering(self):
        print("running clustering model - {}".format(self.clustering_model))
        t1 = t.time()
        self.run_clustering_model()
        print("finished {} clustering in - {:.3f} s".format(self.clustering_model, t.time() - t1))

    def check_cluster_grouping(self):
        t1 = t.time()
        self.app_dict["intents"] = self.app_dict["clusterer"].group_by(
            self.app_dict["model"].labels_, self.min_samples, self.app_dict["output_sentences"])
        print("number of clusters with min {} utterances in each - {}".format(self.min_samples, len(self.app_dict["intents"].keys())))

    def check_slots(self):
        t1 = t.time()
        self.identify_possible_slots()
        print("finished identifying synonyms in - {:.3f} s".format(t.time() - t1))

    def execute_step(self, step):
        switcher = {
            "preprocess": lambda : self.loop_steps(),
            "slots": lambda : self.check_slots(),
            "embed": lambda : self.check_embedding(),
            "cluster": lambda : self.check_clustering(),
            "intents": lambda : self.check_cluster_grouping()
        }
        return switcher.get(step, lambda: "invalid action")()

    def execute(self, return_type):
        self.return_type = return_type
        for step in execution_steps:
            print("******** running the step - {} ********".format(step))
            t1 = t.time()
            self.execute_step(step)
            print("******** finished the step - {} in - {:.3f} s ********".format(step, t.time() - t1))

            if step == "preprocess":
                if len(self.app_dict["step_output"]) == len(self.app_dict["output_sentences"]):
                    print("no of utterances post preprocessing is {}".format(len(self.app_dict["step_output"])))
                else:
                    print("preprocessed utterances are not matching te output utterances counts")
                    self.finalise()
                    return {status: 500, data: "", message: "something went wrong"}

            if step == "embed":
                if self.app_dict["vectors"] is None or len(self.app_dict["vectors"]) <= 0:
                    print("something went wrong, vectors is empty")
                    self.finalise()
                    return {status: 500, data: "", message: "something went wrong"}

            if step == 'cluster':
                if self.app_dict["model"] is None:
                    print("something went wrong, clustering model is not created")
                    self.finalise()
                    return {status: 500, data: "", message: "semething went wrong"}

            if step == "intents":
                if self.app_dict["intents"] is None or len(self.app_dict["intents"]) <= 0:
                    print("something went wrong, intents are empty")
                    self.finalise()
                    return {status: 500, data: "", message: "something went wrong"}

            if return_type == "slots" and step == "slots":
                self.finalise()
                return {"status": 200, "message": "success", "data": {"process_id": self.process_id, "identified_slots": self.synonyms}}
            elif return_type == "intents" and step == "intents":
                intnts = self.app_dict["intents"]
                self.finalise()
                return {'status': 200, 'message': "success", 'data': {"process_id": self.process_id, "intents": intnts}}

    def finalise(self):
        self.app_dict.clear()
        self.excel_data = None

