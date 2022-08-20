import argparse
import itertools
import json
import pickle
import re
from multiprocessing import Process, Manager

import numpy as np
import pandas as pd
import spacy
from gensim.models import FastText
from nltk.stem import PorterStemmer
from sklearn.decomposition import KernelPCA
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

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


def prepare_corpus(in_corpus, processing_chunk, ingredient_oriented):
    json_fail_count_temp = 0
    json_fail_temp = []
    for ingredients_entities in tqdm(processing_chunk):
        sentence = []
        try:
            for entity in json.loads(ingredients_entities):
                if ingredient_oriented and not entity["type"] == "FOOD":
                    continue
                sentence.append("_".join(normalize_name(entity["entity"].replace("-", "_")).split(" ")))
            in_corpus.append(" ".join(sentence).lower())
        except:
            json_fail_temp.append(ingredients_entities)
            json_fail_count_temp += 1


def pmi_ingredient_oriented(df):
    '''
    Calculate the positive pointwise mutal information score for each entry
    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    We use the log( p(y|x)/p(y) ), y being the column, x being the row
    '''
    # Get numpy array from pandas df
    arr = df.values
    # arr = csr_array(arr)
    arr = arr.astype(np.float32)
    # row_totals = arr.sum(axis=1).astype(np.float32)

    ingredients_number = arr.shape[0]

    # p(x,y)/(p(x)*p(y)) = pmi

    # p(x,y)
    arr_symmetric = np.zeros((ingredients_number, ingredients_number))
    for doc_term in tqdm(arr.transpose()):
        non_zero_idx = np.nonzero(doc_term)[0]
        for pair in itertools.combinations(non_zero_idx, 2):
            arr_symmetric[pair[0], pair[1]] += 1
            arr_symmetric[pair[1], pair[0]] += 1
    row_totals = arr_symmetric.sum(1).astype(np.int32)

    # normalizing p(x,y)
    arr_symmetric = arr_symmetric / row_totals.sum()

    px = arr_symmetric.sum(0)

    # p(y)
    arr_symmetric[arr_symmetric == 0] = 0.0001
    px[px == 0] = 0.0001

    arr_symmetric = np.log(arr_symmetric)
    arr_symmetric = arr_symmetric.transpose() - np.log(px)

    _pmi = arr_symmetric - np.log(px)
    _pmi[_pmi < 0] = 0

    _pmi[np.isnan(_pmi)] = 0.00001
    return _pmi


def pmi_recipe_oriented(df):
    '''
    Calculate the positive pointwise mutal information score for each entry
    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    We use the log( p(y|x)/p(y) ), y being the column, x being the row
    '''
    # Get numpy array from pandas df
    arr = df.values
    # p(y|x) probability of each t1 overlap within the row
    # row_totals = sum_numba_parallel(arr)
    row_totals = arr.sum(axis=1)
    # row_totals = ne.evaluate("np.sum(arr, axis=1)")
    prob_cols_given_row = (arr.T.astype(np.float32) / row_totals.astype(np.float32)).T

    # p(y) probability of each t1 in the total set
    col_totals = arr.sum(axis=0)
    prob_of_cols = col_totals / sum(col_totals)
    # PMI: log( p(y|x) / p(y) )
    # This is the same data, normalized
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio == 0] = 0.00001

    ratio[np.isnan(ratio)] = 0.00001
    _pmi = np.log(ratio)
    _pmi[_pmi < 0] = 0
    _pmi[np.isnan(_pmi)] = 0.00001
    return _pmi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate the pmi based vector embeddings')
    parser.add_argument("--dataset_path", type=str, default="dataset/dataset.csv")
    parser.add_argument("--ingredient_oriented", type=bool, default=False)
    parser.add_argument("--model_type", type=str, default="PMI", help='You can choose either "PMI" or "FastText"')
    parser.add_argument("--jobs_n", type=int, default=24)
    parser.add_argument("--output_name", type=str, default="ingredeint2vector")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    ingredient_oriented = args.ingredient_oriented
    jobs_n = args.jobs_n
    output_name = args.output_name
    model_type = args.model_type.lower()
    if not model_type.lower() in ["pmi", "fasttext"]:
        print(f"You have chosen wrong model_type: {args.model_type}. Choose either PMI or FastText")

    # load dataset. First download from drive
    # https://drive.google.com/drive/u/0/folders/1gkaAL3ebbMsxP_IBqGdqfUd_Z5pYMKfW
    df = pd.read_csv(dataset_path, error_bad_lines=False, encoding='latin-1')

    corpus = []

    list_to_process = df["ingredients_entities"].tolist()
    chunk_n = int(len(list_to_process) / jobs_n)

    # prepare the corpus of food entities
    with Manager() as manager:
        L = manager.list()  # <-- can be shared between processes.
        processes = []
        for i in tqdm(range(jobs_n)):
            if i == jobs_n - 1:
                p = Process(target=prepare_corpus, args=(L, list_to_process[i * chunk_n:], ingredient_oriented))
            else:
                p = Process(target=prepare_corpus,
                            args=(L, list_to_process[i * chunk_n: (i + 1) * chunk_n], ingredient_oriented))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        corpus = list(L)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    ingredeint2vector = {}

    if model_type == "pmi":
        dtm = X.toarray()

        # calculate the pmi of the inverse of dtm
        tdm = dtm.transpose()
        pmi_tdm = {}
        if ingredient_oriented:
            pmi_tdm = ingredient_oriented(pd.DataFrame(tdm))
        else:
            pmi_tdm = pmi_recipe_oriented(pd.DataFrame(tdm))

        pca = KernelPCA(n_components=300, n_jobs=jobs_n)
        pmi_tdm_reduced = pca.fit_transform(pmi_tdm)

        ingredeint2vector = {vectorizer.get_feature_names()[idx]: elem for idx, elem in enumerate(pmi_tdm_reduced)}

    if model_type == "fasttext":
        fasttext_model = FastText(vector_size=300, window=10, min_count=2)
        fasttext_model.build_vocab(corpus_iterable=corpus)
        fasttext_model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=5)  # train

        ingredeint2vector = {name: fasttext_model.wv[name] for name in vectorizer.get_feature_names() if
                             name in fasttext_model.wv}
    with open(f'{output_name}.pickle', 'wb') as handle:
        pickle.dump(ingredeint2vector, handle, protocol=pickle.HIGHEST_PROTOCOL)
