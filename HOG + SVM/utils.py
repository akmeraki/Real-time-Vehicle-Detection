import matplotlib.image as mpimg
from skimage.feature import hog
from os import walk, path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import cv2
import numpy as np



### PARAMETERS

# HOG

orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block, which can handel e.g. shadows
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins


# SEARCH REGION AND SLIDING WINDOW

y_start_stop = [400, 720] # Min and max in y to search in slide_window()
ystart_0 = y_start_stop[0]
ystop_0 = ystart_0 + 64*2
ystart_1 = ystart_0
ystop_1 = y_start_stop[1]
ystart_2 = ystart_0
ystop_2 = y_start_stop[1]
ystarts = [ystart_1, ystart_2]
ystops = [ystop_1-100, ystop_2]
search_window_scales = [1.5, 2]  # (64x64), (96x96), (128x128)


def get_file_names(rootdir):
    data=[]
    for root, dirs, files in walk(rootdir, topdown=True):
        for name in files:
            _, ending = path.splitext(name)
            if ending != ".jpg" and ending != ".jepg" and ending != ".png":
                continue
            else:
                data.append(path.join(root, name))
    return data


# Define a function to return HOG features
def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True, feature_vector=feature_vec)

        return features


def bin_spatial(img, size=(32, 32)):

    c1 = cv2.resize(img[:,:,0], size).ravel()
    c2 = cv2.resize(img[:,:,1], size).ravel()
    c3 = cv2.resize(img[:,:,2], size).ravel()
    
    return np.hstack((c1, c2, c3))


# Define a function to compute color histogram features
def color_hist(img, nbins=32):

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


# Define a function to extract features from a list of images
def extract_features(imgs, spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2):

    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        file_features = []

        image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        features.append(single_img_features(image))
    
    return features


# Define a function to extract features from a single image window
def single_img_features(img, spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2):
    # Define an empty list to receive features
    img_features = []

    # Apply color conversion if other than 'RGB'
    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # Compute spatial features 
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    img_features.append(spatial_features)

    # Compute histogram features 
    hist_features = color_hist(feature_image, nbins=hist_bins)
    img_features.append(hist_features)

    # Compute HOG features 
    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.extend(get_hog_features(feature_image[:,:,channel],
                            orient, pix_per_cell, cell_per_block, feature_vec=True))
    img_features.append(hog_features)

    return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, spatial_size=(32, 32), hist_bins=32, orient=9,
                    pix_per_cell=8, cell_per_block=2):

    # Create an empty list to receive positive detection windows
    detected_windows = []
    
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))  # training image is (64,64)

        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block)

        # Scale extracted features to be fed to classifier
        X = np.array(features).reshape(1, -1)
        test_features = scaler.transform(X)

        # Predict using your classifier
        prediction = clf.predict(test_features)

        # If positive (prediction == 1) then save the window    
        if prediction == 1:
            detected_windows.append(window)
    
    return detected_windows


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    detected_windows = []
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]

    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above, hold the number of hog cells
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step: there are 8 cells, and move 2 cells per step, 75% overlap
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    ims = img.copy()
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
        
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))


            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)

            hist_features = color_hist(subimg, nbins=hist_bins)

            X = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(X)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                detected_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    
    return detected_windows