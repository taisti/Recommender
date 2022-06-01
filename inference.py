import pickle
with open('ingredient2ingredients.pickle', 'rb') as handle:
    ingredient2ingredients = pickle.load(handle)

if __name__ == "__main__":
    ingredient = "chicken"
    ingredient_idx = [idx for idx, elem in enumerate(ingredient2ingredients.keys()) if ingredient == elem][0]
    print(ingredient2ingredients[list(ingredient2ingredients.keys())[ingredient_idx]])

