import cv2
from cv2 import dnn_superres

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread("/Users/ashtonmaze/Code/GitHub/T3/python/img/2.png", cv2.IMREAD_GRAYSCALE)

# Read the desired model
# path = "/Users/ashtonmaze/Code/GitHub/T3/python/FSRCNN_x2.pb"
path = "/Users/ashtonmaze/Code/GitHub/T3/python/EDSR_x4.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 3)
#sr.setModel("fsrcnn", 2)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite("./upscaled.png", result)