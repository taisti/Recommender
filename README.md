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
5. Optionally perform inference by running the ``inference.py --embedding_path <path to the ingredient embeddings dictionary> --source_token <name of the ingredient for which you want to find the most similar substitutes> --topn <number of substitutes you want to find for given ingredient>``
