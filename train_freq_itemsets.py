import argparse
import json
import pickle
import re
from multiprocessing import Process, Manager

import pandas as pd
import spacy
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from nltk.stem import PorterStemmer
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


def prepare_corpus(in_corpus, processing_chunk):
    json_fail_count_temp = 0
    json_fail_temp = []
    for ingredients_entities in tqdm(processing_chunk):
        sentence = []
        try:
            for entity in json.loads(ingredients_entities):
                if entity["type"] == "FOOD":
                    sentence.append("_".join(normalize_name(entity["entity"].replace("-", "_")).split(" ")))
            in_corpus.append(" ".join(sentence).lower())
        except:
            json_fail_temp.append(ingredients_entities)
            json_fail_count_temp += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate the frequent itemsets')
    parser.add_argument("--dataset_path", type=str, default="dataset/dataset.csv")
    parser.add_argument("--output_name", type=str, default="fequent_itemsets")
    parser.add_argument("--jobs_n", type=int, default=24)

    args = parser.parse_args()

    jobs_n = args.jobs_n
    dataset_path = args.dataset_path
    output_name = args.output_name
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
                p = Process(target=prepare_corpus, args=(L, list_to_process[i * chunk_n:]))
            else:
                p = Process(target=prepare_corpus,
                            args=(L, list_to_process[i * chunk_n: (i + 1) * chunk_n]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        corpus = list(L)

    dataset = [elem.split() for elem in corpus]

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True, verbose=1)

    with open(f"{output_name}.pickle", 'wb') as handle:
        pickle.dump(frequent_itemsets, handle, pickle.HIGHEST_PROTOCOL)
