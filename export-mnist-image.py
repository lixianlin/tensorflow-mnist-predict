from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy
import os

data_sets = input_data.read_data_sets('D:/tensorflow/mnist/input_data')

images = data_sets.train.images
labels = data_sets.train.labels

total = images.shape[0]

save_path = 'D:/tensorflow/mnist-image'

for i in range(0, total):
    label = labels[i]

    # create path if not exists
    path = os.path.join(save_path, str(label))
    if not os.path.exists(path):
        os.makedirs(path)

    name = str(i) + '.bmp'
    filename = os.path.join(path, name)

    # skip if file exists
    if os.path.exists(filename):
        print(filename, 'exists')
        continue

    # restore and save image
    image = images[i]
    image = image.reshape(28, 28)
    image = numpy.multiply(image, 255)
    image = image.astype(numpy.uint8)

    im = Image.fromarray(image)
    im.save(filename)
    print(filename, 'saved')

