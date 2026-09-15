from PIL import Image

import numpy as np

IMG_SIZE = (299, 299)

def preprocess(img_path):

img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)

x = np.array(img)/255.0

x = x[np.newaxis, ...]

return x

This is the preprocessing code that is done during the image upload