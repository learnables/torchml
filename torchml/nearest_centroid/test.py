from nearest_centroid import *

import numpy as np
#import torchml as ml
samples = torch.from_numpy(np.array([[-1.0,-1.0],[-2.0,-3.0],[2.0,3.0]]))
classes = torch.from_numpy(np.array([-1, -1, 1]))
cent = NearestCentroid()
cent.fit(samples,classes)
print(cent.predict(torch.from_numpy(np.array([-5,-5]))))