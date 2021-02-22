# Automatic Utterances Clustering for Chatbots

*A python model which automatically clean up the utterances, 
generate the possible slots and clusters. Later, 
it will be easy to look and merge the clusters together 
to form as many intents as you need.*

Assuming you have 10,000 utterances and to perform below operations 
would definitely takes a lot of time 
- **Cleaning up the utterances** - This includes removing unwanted utterances, email ids, numerics and special characters
- **Create Intents** - Analyse the utterances and start clustering the related utterances together to form intents
- **Identify Possible Slots** - Identify related words in the utterances to form slots/entities

So just run this algorithm to ease your work 
(there will be certain amount of effort you need
to put from your end to finish your work).

### So, how to use/run this code?
1. Download or clone the master branch
2. ```$ pip install -r requirements.txt -t . ```
(This installs all the python package 
dependencies required to run this code)
3. Save and run the application using the command
```python test1.py```
4. Find for the output excel file in the same directory where handler.py exists

-----------------------------------------------

If you have time and would like to experiment with the advance settings (are optional). 
You can try modifying the values and see the differences.

##### Input Parameters in handler.py
```JSON format
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
```

1. **botname** - Give any alphanumeric+underscore name without space
2. **excel_data** - Pass your excel file path which contains only utterances
3. **synonyms_generating_type** - These are the ways of generating the synonyms. Choose one as per your need.
    - auto - Selecting this, the model looks for the pattern of words appearing 
    - custom - If you already have synonyms in the utterances to find synonyms on its own
4. **custom_synonyms** - This is required if you have selected "custom_synonyms" in synonyms_generating_type
else leave it blank. This take the dictionary format of input as below
    ```
    {
    "action":["add", "remove", "register", "signup", "cancel", "delete", "updated"],
    "fruits":["orange","apple","pineapple"]
    }
    ```

5. **auto_generate_synonyms_modes** - It has 3 different modes based on which it looks at the utterances 
for similar words to create slots. Choose one among the below - 
    - strict
    - moderate
    - loose
6. **remove_unimportant_word** - If you have already know there are some unwanted words 
in the utterances and you do not want them in the process of clustering then, 
you can list here. It will definitely help the model to cluster in better way.
You can also list the words which are common in every utterences.
7. **output_utterances_type** - 
    - alphanumeric : If your clustered output to be in alphanumeric
    - extract_only_text : If you want to filter all unncessary things and 
    just want only plain text in the clustered output
8. **min_samples** - What is the minimum number of utterances you need in each cluster
9. **embedding_model** - Choose the embedding model from one of these - "tfidf", "word2vec", "w2v_tfidf", "bert"
10. **clustering_model** - Choose the clustering model from one of these - "custom", "ahc", "kmeans", "hdbscan"
11. **no_of_clusters** - This option will be ignored if you are choose the clustering model as "custom" or "hdbscan", else you need to specify how many clusters do you need.
12. **merge_similar_clusters** - This value is considered if your clustering model is "custom". Initially the custom model try to cluster maximum clusters possible and try to see if there are any clusters which can be merged togather. If 80 is specified, then model tries to merge all those clusters which are greater than or equal to 80 percent similar.
13. **ignore_similarity** - This value is considered if your clustering model is "custom". Ignore the clusters whose utterances are lesser than the given percentage
