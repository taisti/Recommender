import argparse
import pickle
import random

from sklearn.metrics.pairwise import cosine_similarity


def generate_ingredient2ingredients(ingredeint2vector, token):
    source_vec = ingredeint2vector[token]
    ingredients_scores = {ingr: cosine_similarity(ingredeint2vector[ingr].reshape(1, -1), source_vec.reshape(1, -1))
                          for ingr in ingredeint2vector.keys() if ingr != token}
    sorted_ingredients_scores = {k: v for k, v in
                                 sorted(ingredients_scores.items(), key=lambda item: item[1], reverse=True)}
    return sorted_ingredients_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get top n entities given embeddint dictionary')
    parser.add_argument("--model_path", type=str, default="ingredeint2vector.pickle",
                        help="Pass the path to the ingredient embeddings dictionary")
    parser.add_argument("--source_token", type=str, default="chicken",
                        help='Enter the name of the ingredient for which you want to find the most similar substitutes')
    parser.add_argument("--topn", type=int, default=10,
                        help='Enter the number of substitutes you want to find for given ingredient')
    parser.add_argument("--model_type", type=str, default="FastText",
                        help='Enter the type of the model: either "Embedding" or "Apriori"')

    args = parser.parse_args()

    embedding_path = args.embedding_path
    source_substitute = args.source_token
    topn = args.topn
    model_type = args.model_type
    with open(embedding_path, 'rb') as handle:
        model = pickle.load(handle)

    if model_type == "Embedding":
        sorted_ingredients_scores = generate_ingredient2ingredients(model, source_substitute)
        print(f"The top {topn} substitutes for given ingredient: {source_substitute}:")
        print({k: sorted_ingredients_scores[k] for k in list(sorted_ingredients_scores)[:topn]})
    else:
        len_fa = 5
        used_contexts = []
        freq_items_for_source_ingr = [(
            model["support"][model["itemsets"][model["itemsets"] == el].index[0]],
            el) for el in model['itemsets'] if
            {source_substitute}.issubset(el) and len(el) == len_fa and (
                not any([ee for ee in used_contexts if ee.issubset(el)]))]
        freq_items_for_source_ingr.sort(key=lambda x: x[0], reverse=True)
        select_idx = random.randint(0, 3)
        context_ingredients = set(freq_items_for_source_ingr[select_idx][1])
        context_ingredients.remove(source_substitute)
        used_contexts.append(context_ingredients)
        freq_items_context_ingr = [(model["support"][
                                        model["itemsets"][model["itemsets"] == el].index[0]],
                                    el)
                                   for el in
                                   model['itemsets'] if
                                   context_ingredients.issubset(el) and len(el) == len_fa and not {
                                       source_substitute}.issubset(el)]
        freq_items_context_ingr.sort(key=lambda x: x[0], reverse=True)
        res = []
        for count in range(topn):
            ss2 = set(freq_items_context_ingr[count][1])
            for el in list(context_ingredients):
                ss2.remove(el)
            res.append(ss2)
        print(source_substitute, res)
