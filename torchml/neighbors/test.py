from nearest_centroid import *
from sklearn.neighbors import NearestCentroid as sk

import numpy as np
#import torchml as ml
X = np.array([[-1,-1],[-2,-3],[2,3]])
y = np.array([-1, -1, 1])
samples = torch.from_numpy(X)
classes = torch.from_numpy(y)
Tcent = NearestCentroid()
clf = sk()
Tcent.fit(samples,classes)
clf.fit(X,y)
parray = np.array([[2,2]])
print("clf: ",clf.predict(parray))
print("Tcent: ",Tcent.predict(torch.from_numpy(parray)).numpy())