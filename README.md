# mlfacetoname
This project aims to predict a name given a face. The current predictions are based on the Labeled Faces of Wild (LFW) dataset. The predictions can be easily improved by using more datasets (use "calculate_lfw_embeddings.py" for that).

## Predict a name
Copy the Jupyter notebook "vgg_net_tensorflow" and the pickle file "dict_lfw.pkl" from this repository. Then on the bottom of the Jupyter notebook simply add the file names of pictures of people whose name you want to predict using this model.

## How it works
Using a pretrained VGG NET model the embeddings of the LFW dataset have been calculated. Then the embeddings of people with the same first name have been averaged which then represents the vector of a face associated with this first name. If one inputs a new face, then the embedding is calculated and the closest precalculated vector is found which is then used as name prediction.
