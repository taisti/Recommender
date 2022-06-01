import zipfile
import os
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import re
import spacy
from nltk.stem import PorterStemmer
from multiprocessing import Process, Manager

nlp = spacy.load("en_core_web_lg")
stopwords = nlp.Defaults.stop_words
ps = PorterStemmer()


def normalize_name(name):
    name = re.sub(r'[^a-zA-Z]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.lower()
    name = ' '.join([token for token in name.split(' ') if token not in stopwords])
    stemmed_name = " ".join([ps.stem(token.text) for token in nlp(name)])
    return stemmed_name


def prepare_corpus(corpus, processing_chunk):
    json_fail_count_temp = 0
    json_fail_temp = []
    for ingredients_entities in tqdm(processing_chunk):
        sentence = []
        try:
            # print(json.loads(ingredients_entities))
            for entity in json.loads(ingredients_entities):
                if entity["type"] == "FOOD":
                    sentence.append("_".join(normalize_name(entity["entity"].replace("-", "_")).split(" ")))
            corpus.append(" ".join(sentence).lower())
        except:
            json_fail_temp.append(ingredients_entities)
            json_fail_count_temp += 1


def pmi(df):
    '''
    Calculate the positive pointwise mutal information score for each entry
    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    We use the log( p(y|x)/p(y) ), y being the column, x being the row
    '''
    # Get numpy array from pandas df
    arr = df.values

    # p(y|x) probability of each t1 overlap within the row
    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = np.divide(arr.T / row_totals).T

    # p(y) probability of each t1 in the total set
    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)

    # PMI: log( p(y|x) / p(y) )
    # This is the same data, normalized
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio == 0] = 0.00001
    ratio[np.isnan(ratio)] = 0.00001
    _pmi = np.log(ratio)
    _pmi[_pmi < 0] = 0

    return _pmi


def generate_ingredient2ingredients(ingredient2ingredients, processing_chunk):  # the managed list `L` passed explicitly.
    for ingredient, vector in tqdm(processing_chunk):
        ingredients_scores = [tuple([ingr, cosine_similarity(vector.reshape(1, -1), vec.reshape(1, -1))]) for ingr, vec
                              in ingredeint2vector if ingr != ingredient]
        ingredient2ingredients[ingredient] = sorted(ingredients_scores, key=lambda x: x[1], reverse=True)[:20]


if __name__ == "__main__":

    # load dataset. First downlaod from drive
    # https://drive.google.com/drive/u/0/folders/1gkaAL3ebbMsxP_IBqGdqfUd_Z5pYMKfW
    dataset_fname = "dataset/dataset.csv"
    if not os.path.isfile(dataset_fname):
        path_to_zip_file = "dataset/dataset.zip"
        directory_to_extract_to = "./"
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

    # specify number of rows to process
    df = pd.read_csv("dataset/dataset.csv", nrows=100000)

    # specify number of threads to use during computing
    jobs_n = 16

    corpus = []
    json_fail_count = 0
    json_fail = []
    list_to_process = df["ingredients_entities"].tolist()
    chunk_n = int(len(list_to_process) / jobs_n)

    # prepare the corpus of food entities
    with Manager() as manager:
        L = manager.list()  # <-- can be shared between processes.
        processes = []
        for i in tqdm(range(jobs_n)):
            if i == jobs_n - 1:
                p = Process(target=prepare_corpus, args=(L, list_to_process[i * chunk_n:]))
            else:
                p = Process(target=prepare_corpus, args=(L, list_to_process[i * chunk_n: (i + 1) * chunk_n]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        corpus = list(L)

    # save the prepared corpus
    with open('corpus.pickle', 'wb') as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # create the document term matrix (dtm) as needed for later processing
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    dtm = np.array(X.toarray())

    # calculate the pmi of the inverse of dtm
    tdm = dtm.transpose()
    pmi_tdm = pmi(pd.DataFrame(tdm))

    # save the pmi_tdm
    with open('pmi_tdm.pickle', 'wb') as handle:
        pickle.dump(pmi_tdm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # reduce the dimensionality before using
    pca = KernelPCA(n_components=999, n_jobs=jobs_n)
    pmi_tdm_reduced = pca.fit_transform(pmi_tdm)

    ingredeint2vector = [tuple([vectorizer.get_feature_names()[idx], elem]) for idx, elem in enumerate(pmi_tdm_reduced)]

    ingredient2ingredients = {}
    chunk_n = int(len(ingredeint2vector) / jobs_n)

    #generate ingredient2ingredients dictionary - the dictionary of 20 most simmilar ingredients for each ingredient based on the pmi dtm and cosine similarity
    with Manager() as manager:
        L = manager.dict()  # <-- can be shared between processes.
        processes = []
        for i in tqdm(range(jobs_n)):
            if i == jobs_n - 1:
                p = Process(target=generate_ingredient2ingredients, args=(L, ingredeint2vector[i * chunk_n:]))
            else:
                p = Process(target=generate_ingredient2ingredients, args=(L, ingredeint2vector[i * chunk_n: (i + 1) * chunk_n]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        ingredient2ingredients = dict(L)

    #save the ingredient2ingredients dict
    with open('ingredient2ingredients.pickle', 'wb') as handle:
        pickle.dump(ingredient2ingredients, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #check the 'chicken' ingredient for most simmilar ingredients
    ingredient = "chicken"
    ingredient_idx = [idx for idx, elem in enumerate(ingredient2ingredients.keys()) if ingredient == elem][0]
    print(ingredient2ingredients[list(ingredient2ingredients.keys())[ingredient_idx]])
