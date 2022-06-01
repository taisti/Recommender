# The repository contains implementations of different recommenders that is used for ingredients substitutions

## BASELINE

The baseline recommender is implemented according to the veganizer paper (no link to the publication). It is based on computing on the Positive Pointwise Mutual Information from the DTM matrix, where columns denote ingredients and rows denote recipes (if an ingredient occurs in a recipe, it has a value of 1, otherwise 0). 

# Baseline usage
1. Copy the repository.
2. Install the requirements.
3. Download the dataset from the google drive https://drive.google.com/drive/u/0/folders/1gkaAL3ebbMsxP_IBqGdqfUd_Z5pYMKfW and extracted to the project directory.
4. Train the model:
    ``python train_baseline.py``

      The ``ingredient2ingredents`` dict will be created and saved in pickle format in the project directory. This is a very basic dictionary that returns the 20 most similar components for the given component key. 
5. Optionally perform inference by running the ``inference.py`` file - the ``ingredient2ingredents`` dictionary must have been previously created. Change the value of the ``ingredient`` variable in the ``inference.py`` file to search for other ingredient replacements.
