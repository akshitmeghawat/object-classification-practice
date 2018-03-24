import numpy as np
from sklearn import mixture
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Read Image
img1 = Image.open("ilk-3b-1024.tif")
img = np.array(img1)

# take a subset
crop_img = img[0:500, 0:500]

plt.imshow(crop_img)
plt.show()

X = crop_img.reshape((-1, 1))
X1 = img.reshape((-1, 1))

print 'image shape after cropping', crop_img.shape
print 'shape of X', X.shape

n_components = range(2, 10)
for n in n_components:
    g = mixture.GaussianMixture(n_components=n, covariance_type='full')
    g.fit(X)
    print '# of Observations:', X.shape
    print 'Gaussian Weights:', g.weights_
    print 'Gaussian Means:', g.means_

    crop_img_clustered = g.predict(X)
    crop_img_clustered.shape = crop_img.shape
    crop_img_clustered = (crop_img_clustered.astype('uint8')) * 255

    img_clustered = g.predict(X1)
    img_clustered.shape = img.shape
    img_clustered = (img_clustered.astype('uint8')) * 255
    img_name = 'gmm_img_' + str(n) + '.png'
    cv2.imwrite(img_name, img_clustered)
