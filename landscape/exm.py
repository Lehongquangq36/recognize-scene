import os
import cv2
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import mahotas

# link data
city_path = './data/city'
forest_path = './data/forest'
sea_path = './data/sea'
# link image
city_image_path = [os.path.join(city_path,i) for i in os.listdir(city_path)]
forest_image_path = [os.path.join(forest_path,i) for i in os.listdir(forest_path)]
sea_image_path = [os.path.join(sea_path,i) for i in os.listdir(sea_path)]

size = 100
test_image_path = city_image_path[480:] + forest_image_path[480:] + sea_image_path[480:]
train_image_path = city_image_path[:size] + forest_image_path[:size] + sea_image_path[:size]


# # path to output
# output_path = "D:\\project\\fruit-classification\\output\\"

# # path to training data
# train_path = "D:\\project\\fruit-classification\\dataset\\train\\"

# # get the training labels
# train_labels = os.listdir(train_path)
# train_labels.sort()

# # num of images per class
# images_per_class = 400

# # fixed-sizes for image
# fixed_size = tuple((100, 100))

# # bins for histogram
bins = 2

# # empty lists to hold feature vectors and labels
# global_features = []
# labels = []

# feature-descriptor-1: Hu Moments hình dạng
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture kết cấu, hoa văn
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.show()
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    # hist = cv2.calcHist([image],[0],None,[256],[0,256])
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    #print( hist)
    #print( hist.shape)
    # normalize the histogram
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)  
    # return the histogram
    return hist.flatten()
def read_image(image_path):
    return cv2.imread(image_path)
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # return cv2.resize(image,(HEIGHT,WIDTH),interpolation = cv2.INTER_CUBIC)
#print(fd_histogram( read_image(train_image_path[0])))
#print(len(fd_histogram( read_image(train_image_path[0]))))
img = read_image(train_image_path[0])
xkernel = np.array([[-1, 0, 1]])
ykernel = np.array([[-1], [0], [1]])
dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
print(img.shape)
print(dx.shape)
# fd_histogram( read_image(train_image_path[0]))
# read image form each folder

# # loop over the training data sub-folders
# for training_name in train_labels:
#     # join the training data path and each species training folder
#     dir = os.path.join(train_path, training_name)

#     # get the current training label
#     current_label = training_name
#     # loop over the images in each sub-folder
#     for x in range(1,images_per_class+1):
#         # get the image file name
#         file = dir + "\\" + "Image ("+str(x) + ").jpg"

#         print(file)
#         # read the image and resize it oto a fixed-size
#         image = cv2.imread(file)
#         image = cv2.resize(image, fixed_size)

#         ####################################
#         # Global Feature extraction
#         ####################################
#         fv_hu_moments = fd_hu_moments(image)
#         fv_haralick   = fd_haralick(image)
#         fv_histogram  = fd_histogram(image)

#         ###################################
#         # Concatenate global features
#         ###################################
#         global_feature = np.hstack([fv_histogram, fv_hu_moments, fv_haralick])

#         # update the list of labels and feature vectors
#         labels.append(current_label)
#         global_features.append(global_feature)

#     print("[STATUS] processed folder: {}".format(current_label))

# print("[STATUS] completed Global Feature Extraction...")


# # get the overall feature vector size
# print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# # get the overall training label size
# print ("[STATUS] training Labels {}".format(np.array(labels).shape))

# # encode the target labels
# le = LabelEncoder()
# target = le.fit_transform(labels)

# # normalize the feature vector in the range (0-1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# rescaled_features = scaler.fit_transform(global_features)

# # save the feature vector using HDF5
# h5f_data = h5py.File(output_path+'data.h5', 'w')
# h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

# h5f_label = h5py.File(output_path+'labels.h5', 'w')
# h5f_label.create_dataset('dataset_1', data=np.array(target))

# h5f_data.close()
# h5f_label.close()

# print("[STATUS] end of training..")