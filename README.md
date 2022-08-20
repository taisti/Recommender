# The repository contains implementations of different recommenders that is used for ingredients substitutions

## BASELINE

The baseline recommender is implemented according to the veganizer paper (no link to the publication). It is based on computing on the Positive Pointwise Mutual Information from the DTM matrix, where columns denote ingredients and rows denote recipes (if an ingredient occurs in a recipe, it has a value of 1, otherwise 0). 

# Baseline usage
1. Copy the repository.
2. Install the requirements.
3. Download the dataset from the google drive https://drive.google.com/drive/u/0/folders/1gkaAL3ebbMsxP_IBqGdqfUd_Z5pYMKfW and extracted to the project directory.
4. Train the model:

    ``python train_baseline.py --dataset_path <path to the downloaded dataset> --ingredient_oriented <True|False - whether use words classified as FOOD entity or other entities during training> --model_type <PMI|FastText which kind of recommender we want> --jobs_n <number of threads to be used> --output_name <output path to ingredient vectors>``

      The output is dict that contains the ingredient embedding generated with the use of different models in pickle format.  
5. Optionally perform inference by running the 

    ``inference.py --embedding_path <path to the ingredient embeddings dictionary> --source_token <name of the ingredient for which you want to find the most similar substitutes> --topn <number of substitutes you want to find for given ingredient>``

6. If the original dataset is too large for the training, we can split it for N equal sized datasets with the use of:

    ``python divide_dataset.py --dataset_path <path to the original dataset> --n_pieces <number of pieces you want to divide your dataset into>``

7. If the dataset was splitted and for each part was created a sepearate embedding, we can try to inference as the average of all embeddings. For this purpose run: ``inference_multiple_embeddings.py --embedding_dir <the parth to the directory where are stored embeddings. Note - the embeddings should have name "ingredeint2vector<number>.pickle">``, where the number indicates the number of the embedding. If we have 2 embeddings, we should name them: ingredeint2vector0.pickle and ingredeint2vector0.pickle> --source_token <name of the ingredient for which you want to find the most similar substitutes> --topn <number of substitutes you want to find for given ingredient> --datasets_number <number of datasets subsets>``
