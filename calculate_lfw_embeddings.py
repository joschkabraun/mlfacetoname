# oriented at https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/

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

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
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

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
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

# get all the paths of all files (including subdirectories) of given format
def get_paths_of_files_of_format(path, file_format="jpg"):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files.append(os.path.join(r, file))
    
    return files

# extract names of the images in the used files and composes them as (first name) (given name) (i.e. with a space)
def extract_full_names(path_files,dataset="lfw"):
    names = []
    for f in path_files:
        fname = f.split("\\")[-1]              # the files always were separated by "\\" in the end
        
        if dataset=="lfw":
            fname = fname.split("_")
            names.append(fname[0]+" "+fname[1])
        else:
            raise NotImplementedError("Only code for the lfw dataset have been implemented")
    return names

# averages vectors to associated name; expects numpy vectors as input
class NamedAvgVector:
    def __init__(self, name, numpy_vec="a"):
        self.name = name
        self.vec = numpy_vec
        if numpy_vec != "a":
            self.n = 1
        else:
            self.n = 0
    
    def update(self, numpy_vec):
        if self.n != 0:
            self.vec *= self.n
            self.vec += numpy_vec
            self.n += 1
            self.vec /= self.n
        else:
            self.n = 1
            self.vec = numpy_vec
            
# creates a dictionary with all the first names as keys and associated empty NamedAvgVector data structures
def create_initial_dict(full_names):
    dict_list = []
    for name in full_names:
        first_name = name.split(" ")[0]
        dict_list.append((first_name, NamedAvgVector(first_name)))
    return dict(dict_list)

# updates the dictionary of NamedAvgVectors with the embeddings
# assumes that all embeddings are already in dictionary
# i_0_f_names : is the starting point of the calculated embeddings in full_names
def update_dict(d, full_names, i_0_f_names, emb):
    for i in range(len(emb)):
        fname = full_names[i_0_f_names + i].split(" ")[0]
        d[fname].update(np.array(emb[i]))
        
# calculates the averaged embeddings for all the first names given a dataset
def calculate_embeddings(path_dataset, step_size=100, file_format="jpg", dataset_name="lfw"):
    im_paths = get_paths_of_files_of_format(path_lfw, file_format)
    size = len(im_paths)
    full_names = extract_full_names(im_paths)

    #dict_navecs = create_initial_dict(full_names)
    num = 131
    f = open("dict_{}.pkl".format(num), "rb")
    dict_navecs = pickle.load(f)
    f.close()
    
    for i in range(num, math.ceil(size/step_size)):
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
