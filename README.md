# The repository contains implementations of different recommenders that is used for ingredients substitutions

## BASELINE

The repository contains 5 different baseline recommenders:
1. Positive Pointwise Mutual Information (PMI) Ingredient Oriented
2. Positive Pointwise Mutual Information Recipe Oriented
3. FastText Recipe Oriented
4. FastText Ingredient Oriented
5. Apriori based 

# Baseline usage
1. Copy the repository.
2. Install the requirements.
3. Download the dataset from the google drive https://drive.google.com/drive/u/0/folders/1gkaAL3ebbMsxP_IBqGdqfUd_Z5pYMKfW and extracted to the project directory.
4. Train the model (Each PMI and FastText)

    ``python train_baseline.py --dataset_path <path to the downloaded dataset> --ingredient_oriented <True|False - whether use words classified as FOOD entity or other entities during training> --model_type <PMI|FastText which kind of recommender we want> --jobs_n <number of threads to be used> --output_name <output path to ingredient vectors>``

5. Train the Apriori based model
 ``python train_frequent_itemstes.py --dataset_path <path to the dataset> --output_name <path to the output frequent itemset model> --jobs_n <number of threads available>`` 
 
6. Inference

    ``inference.py --model_path <path to the trained model/embedding (PMI,FastText,Apriori)>  --model_type <type of the entered model (Embedding or Apriori)> --source_token <name of the ingredient for which you want to find the most similar substitutes> --topn <number of substitutes you want to find for given ingredient>``

7. If the original dataset is too large for the training, we can split it for N equal sized datasets with the use of:

    ``python divide_dataset.py --dataset_path <path to the original dataset> --n_pieces <number of pieces you want to divide your dataset into>``

8. If the dataset was splitted and for each part was created a sepearate embedding, we can try to inference as the average of all embeddings. For this purpose run: 
 
    ``inference_multiple_embeddings.py --embedding_dir <the path to the directory where are stored embeddings> --source_token <name of the ingredient for which you want to find the most similar substitutes> --topn <number of substitutes you want to find for given ingredient> --datasets_number <number of datasets subsets>``
    
    Note - the embeddings directory should contain embeddings with following name convention "ingredeint2vector<number>.pickle", where the number indicates the number of the embedding. If we have 2 embeddings, we should name them: ingredeint2vector0.pickle and ingredeint2vector0.pickle
    
