import argparse
import pickle
import sys

from sklearn.metrics.pairwise import cosine_similarity


def generate_ingredient2ingredients(ingredeint2vector, ingredient):
    source_vec = ingredeint2vector[ingredient]
    ingredients_scores = {ingr: cosine_similarity(ingredeint2vector[ingr].reshape(1, -1), source_vec.reshape(1, -1))
                          for ingr in ingredeint2vector.keys() if ingr != ingredient}
    return ingredients_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get top n entities given embeddint dictionary')
    parser.add_argument("--embedding_dir", type=str, default="embeddings",
                        help="Pass the path to the splitted ingredient embeddings dictionary")
    parser.add_argument("--source_token", type=str, default="chicken",
                        help='Enter the name of the ingredient for which you want to find the most similar substitutes')
    parser.add_argument("--topn", type=int, default=10,
                        help='Enter the number of substitutes you want to find for given ingredient')
    parser.add_argument("--datasets_number", type=int, default=10,
                        help='Enter the number of datasets subsets')

    args = parser.parse_args()

    embedding_dir = args.embedding_dir
    ingredient = args.source_token
    topn = args.topn
    datasets_number = args.datasets_number
    aggregated_ingredient2vector = {}

    for idx in range(0, datasets_number):
        try:
            with open(f'{embedding_dir}/ingredeint2vector{idx}.pickle', 'rb') as handle:
                ingredeint2vector = pickle.load(handle)
        except:
            print(sys.exc_info()[0])
            continue
        token_vec = ingredeint2vector[ingredient]
        ingredients_scores = generate_ingredient2ingredients(ingredeint2vector, ingredient)

        for ingr in ingredients_scores.keys():
            if ingr in aggregated_ingredient2vector:
                aggregated_ingredient2vector[ingr] += ingredients_scores[ingr]
            else:
                aggregated_ingredient2vector[ingr] = ingredients_scores[ingr]
    sorted_aggregated_ingredient2vector = {k: v for k, v in
                                           sorted(aggregated_ingredient2vector.items(), key=lambda item: item[1],
                                                  reverse=True)}
    print(f"The top {topn} substitutes for given ingredient: {ingredient}:")
    print({k: sorted_aggregated_ingredient2vector[k] for k in list(sorted_aggregated_ingredient2vector)[:topn]})
