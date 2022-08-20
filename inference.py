import argparse
import pickle

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
    parser.add_argument("--embedding_path", type=str, default="ingredeint2vector.pickle",
                        help="Pass the path to the ingredient embeddings dictionary")
    parser.add_argument("--source_token", type=str, default="chicken",
                        help='Enter the name of the ingredient for which you want to find the most similar substitutes')
    parser.add_argument("--topn", type=int, default=10,
                        help='Enter the number of substitutes you want to find for given ingredient')

    args = parser.parse_args()

    embedding_path = args.embedding_path
    source_substitute = args.source_token
    topn = args.topn

    with open(embedding_path, 'rb') as handle:
        ingredient2ingredients = pickle.load(handle)

    sorted_ingredients_scores = generate_ingredient2ingredients(ingredient2ingredients, source_substitute)
    print(f"The top {topn} substitutes for given ingredient: {source_substitute}:")
    print({k: sorted_ingredients_scores[k] for k in list(sorted_ingredients_scores)[:topn]})
