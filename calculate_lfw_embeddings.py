import math
import numpy as np
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import os
import pickle

def extract_face(filename, required_size=(224, 224)):
    """Extracts a single face from a given photograph using MTCNN.

    Parameters
    ----------
    filename : str
        The filename of which the face shall be extracted
    required_size : touple of int, optional
        The size which the extracted face shall have (default is (224, 224))
    """
    
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def get_embeddings(filenames):
    """Given a list of file names it extracts first the face and then computes their embeddings.

    This method uses VGGFace resnet50 model (which version of VGGNet?)
    
    Parameters
    ----------
    filenames : list of str
        The list of file names of which embeddings shall be calculated
    """
    
    faces = []
    
    # extract faces
    for f_name in filenames:
        try:
            faces.append(extract_face(f_name))
        except ValueError:
            print(f_name)
    
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat

def get_paths_of_files_of_format(path, file_format="jpg"):
    """Finds all the paths of files with specified file format in all subdirectories of the specified path.

    Parameters
    ----------
    path : str
        The path of the starting directory
    file_format : str, optional
        The file format of which the all files shall be found (default is "jpg")
    """
    
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files.append(os.path.join(r, file))
    
    return files

def extract_full_names(path_files,dataset="lfw"):
    """Extract the full names of the given files.

    This function was specifically written for the Labeled Faces of the Wild Dataset (LFW).

    Parameters
    ----------
    path_files : str
        The file paths of which the full names shall be extracted
    dataset : str, optional
        The dataset which is examined (default is "lfw")
    """
    
    names = []
    for f in path_files:
        fname = f.split("\\")[-1]              # the files always were separated by "\\" in the end
        
        if dataset=="lfw":
            fname = fname.split("_")
            names.append(fname[0]+" "+fname[1])
        else:
            raise NotImplementedError("Only code for the lfw dataset have been implemented")
    return names

class NamedAvgVector:
    """A class used to combine a name, a vector (the embedding) and the number of used vectors in one data structure.

    Attributes
    ----------
    name : str
        The name
    vec : numpy array of int
        The average vector
    n : int
        The number of used vectors

    Methods
    -------
    update(numpy_vec)
        Updates the average vector and used vectors with the given numpy vector
    """
    
    def __init__(self, name, numpy_vec="a"):
        """
        Parameters
        ----------
        name : str
            The name
        numpy_vec : numpy vector of int, optional
            The initialized average vector (default is "a" to declare it is unitialized)
        """
        
        self.name = name
        self.vec = numpy_vec
        if numpy_vec != "a":
            self.n = 1
        else:
            self.n = 0
    
    def update(self, numpy_vec):
        """Updates the average vector and used vectors with the given numpy vector

        Parameters
        ----------
        numpy_vec : numpy array of int
            The vector with which the average vector shall be updated
        """
        
        if self.n != 0:
            self.vec *= self.n
            self.vec += numpy_vec
            self.n += 1
            self.vec /= self.n
        else:
            self.n = 1
            self.vec = numpy_vec
            
def create_initial_dict(full_names):
    """Returns a dictionary of NamedAvgVector of the first name for the given names with unitialized average vector.

    Parameters
    ----------
    full_names : list of str
        The list of full names for which a dictionary of NamedAvgVector shall be created
    """
    
    dict_list = []
    for name in full_names:
        first_name = name.split(" ")[0]
        dict_list.append((first_name, NamedAvgVector(first_name)))
    return dict(dict_list)

def update_dict(d, full_names, i_0_f_names, emb):
    """Updates the dictionary of NamedAvgVector at the specified position with the given embeddings

    Parameters
    ----------
    d : dict of NamedAvgVector
        The to be updated dictionary of NamedAvgVector
    full_names : list of str
        The list of full names used in the dictionary (possibly have repetitions)
    i_0_f_names : int
        The starting index in full_names where the embeddings were calculated
    emb : list of numpy array of int
        The calculated embeddings which shall be used for the update
    """
    
    for i in range(len(emb)):
        fname = full_names[i_0_f_names + i].split(" ")[0]
        d[fname].update(np.array(emb[i]))
        
def calculate_embeddings(path_dataset, step_size=100, file_format="jpg", dataset_name="lfw"):
    """Calculates the averaged embeddings for all the first names given a dataset

    Parameters
    ----------
    path_dataset : str
        The path of the dataset
    step_size : int, optional
        The number of embeddings which shall be calculated for every update (default is 100)
    file_format : str, optional
        The used file format of the images (default is "jpg")
    dataset_name : str, optional
        The underlying used data set (default is "lfw")
    """
    
    im_paths = get_paths_of_files_of_format(path_lfw, file_format)
    size = len(im_paths)
    full_names = extract_full_names(im_paths)

    dict_navecs = create_initial_dict(full_names)
    
    for i in range(math.ceil(size/step_size)):
        print("calculate_embeddings i:", i)
        im_paths_tmp = im_paths[i*step_size : (i+1)*step_size]
        emb_tmp = get_embeddings(im_paths_tmp)
        update_dict(dict_navecs, full_names, i*step_size, emb_tmp)
        
        # save progress so far
        f = open("dict_{}.pkl".format(i), "wb")
        pickle.dump(dict_navecs, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        

path_lfw = # needs to be filled in
calculate_embeddings(path_lfw)
