from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from utils import get_file_names, get_hog_features, bin_spatial, color_hist, extract_features
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Enter the path to dataset")

args = parser.parse_args()


def train_svm(car_path = args.path + 'vehicles', notcars_path = args.path + 'non-vehicles', clf_path = 'weights.p'): 
    
    # Read in cars and notcars
    cars = get_file_names(car_path)
    notcars = get_file_names(notcars_path)

    # set the sample size
    sample_size = min(len(cars), len(notcars))
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    print('filenames are saved!')

    print('extracting car features...')
    car_features = extract_features(cars)
    print('car features extracted!')
    
    print('extracting noncar features...')
    notcar_features = extract_features(notcars)
    print('noncar features extracted!')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',9,'orientations',8,
        'pixels per cell and', 2,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # Use a linear SVC
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # save classifier
    clf_pickle = {}
    clf_pickle["svc"] = svc
    clf_pickle["scaler"] = X_scaler
    clf_pickle["orient"] = 9
    clf_pickle["pix_per_cell"] = 8
    clf_pickle["cell_per_block"] = 2
    clf_pickle["spatial_size"] = (32,32)
    clf_pickle["hist_bins"] = 32
    clf_pickle["color_space"] = 'YCrCb'

    pickle.dump( clf_pickle, open(clf_path, "wb" ) )
    print("Classifier is written into: {}".format(clf_path))

if __name__ == '__main__':
    train_svm()