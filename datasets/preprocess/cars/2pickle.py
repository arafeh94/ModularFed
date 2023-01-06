import logging
import os

import numpy as np
from PIL import Image

import h5py
from matplotlib import pyplot
from numpy import asarray

from src.data.data_container import DataContainer
from src.data.data_provider import PickleDataProvider
import scipy.io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

mat = scipy.io.loadmat('../../raw/cars/cars_train_annos.mat')
data = mat['annotations'][0]
count = 10
size = []
all_x = []
all_y = []
for item in data:
    line = {
        'x1': item[0].flatten()[0],
        'y1': item[1].flatten()[0],
        'x2': item[2].flatten()[0],
        'y2': item[3].flatten()[0],
        'label': item[4].flatten()[0],
        'path': '../../raw/cars/cars_train/' + item[5].flatten()[0]
    }
    image = Image.open(line['path']).crop((line['x1'], line['y1'], line['x2'], line['y2'])) \
        .resize((224, 224), Image.ANTIALIAS)
    image_x = asarray(image, dtype=int)
    try:
        x = image_x.transpose((2, 1, 0)).flatten().tolist()
        if len(x) != 150528:
            raise Exception('invalid size')
        y = line['label']
        all_x.append(x)
        all_y.append(y)
    except Exception:
        print(line)
sx = asarray(all_x, dtype=int)
sy = asarray(all_y, dtype=int)
dc = DataContainer(sx, sy)
print("saving...")
PickleDataProvider.save(dc, '../../pickles/cars.pkl')

#
# images = "../../raw/cars"
# print("loading...")
# all_x = []
# all_y = []
# errors = 0
# show_count = 100
# for location, folders, files in os.walk(images):
#     for index, file in enumerate(files):
#         if file.startswith("a"):
#             file_path = location + "/" + file
#             try:
#                 image = Image.open(file_path).resize((128, 128), Image.ANTIALIAS)
#                 image_x = asarray(image, dtype=int)
#                 x = image_x.transpose((2, 1, 0)).flatten().tolist()
#                 y = int(file[1:2])
#                 if show_count > 0:
#                     show_count -= 1
#                     pyplot.imshow(image_x)
#                     pyplot.show()
#                 if len(x) != 16384 * 3:
#                     raise Exception('invalid size')
#                 all_x.append(x)
#                 all_y.append(y)
#             except Exception as e:
#                 errors += 1
#                 print(f'{e} - total errors: {errors}')
#             # show the image
# print(f'{len(all_x)}-{len(all_y)}')
# x = asarray(all_x, dtype=int)
# y = asarray(all_y, dtype=int)
# dc = DataContainer(x, y)
# print("saving...")
# PickleDataProvider.save(dc, '../../pickles/cars.pkl')
