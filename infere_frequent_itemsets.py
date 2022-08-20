import argparse
import pickle
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inference for substitute among the frequent itemsets')
    parser.add_argument("--frequent_itemest_path", type=str, default="frequent_itemsets",
                        help="Path to the frequent itemset generate previously")
    parser.add_argument("--source_token", type=str, default="chicken",
                        help='Enter the name of the ingredient for which you want to find the most similar substitutes')
    parser.add_argument("--topn", type=int, default=10,
                        help='Enter the number of substitutes you want to find for given ingredient')

    args = parser.parse_args()

    frequent_itemest_path = args.frequent_itemest_path
    source_token = args.source_token
    topn = args.topn
    len_fa = 5

    with open(f"{frequent_itemest_path}", 'rb') as outp:  # Overwrites any existing file.
        frequent_itemsets = pickle.load(outp)

    used_contexts = []
    freq_items_for_source_ingr = [(
        frequent_itemsets["support"][frequent_itemsets["itemsets"][frequent_itemsets["itemsets"] == el].index[0]],
        el) for el in frequent_itemsets['itemsets'] if
        {source_token}.issubset(el) and len(el) == len_fa and (
            not any([ee for ee in used_contexts if ee.issubset(el)]))]
    freq_items_for_source_ingr.sort(key=lambda x: x[0], reverse=True)
    select_idx = random.randint(0, 3)
    context_ingredients = set(freq_items_for_source_ingr[select_idx][1])
    context_ingredients.remove(source_token)
    used_contexts.append(context_ingredients)
    freq_items_context_ingr = [(frequent_itemsets["support"][
                                    frequent_itemsets["itemsets"][frequent_itemsets["itemsets"] == el].index[0]], el)
                               for el in
                               frequent_itemsets['itemsets'] if
                               context_ingredients.issubset(el) and len(el) == len_fa and not {source_token}.issubset(el)]
    freq_items_context_ingr.sort(key=lambda x: x[0], reverse=True)
    res = []
    for count in range(topn):
        ss2 = set(freq_items_context_ingr[count][1])
        for el in list(context_ingredients):
            ss2.remove(el)
        res.append(ss2)
    print(source_token, res)
