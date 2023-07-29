from google.colab import drive
drive.mount('/content/gdrive')

data_dir = "/content/gdrive/MyDrive/Stone"

import cv2
import os
import pandas as pd


img_size = (64, 64)

features = []
labels = []


for filename in os.listdir(data_dir):

    img = cv2.imread(os.path.join(data_dir, filename))

    img = cv2.resize(img, img_size)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    img_edges = cv2.Canny(img_gray, 100, 200)

    img_edges = img_edges / 255.0

    feature = img_edges.flatten()

    features.append(feature)
    labels.append(filename.split('.')[0])

df = pd.DataFrame(features, columns=[f'pixel_{i}' for i in range(len(features[0]))])
df['label'] = labels

df.to_csv('z.csv', index=False)

import pandas as pd


dataset_path = "/content/z.csv"
df = pd.read_csv(dataset_path)


df.loc[df['label'].str.contains('Stone', case=False, na=False), 'label'] = 'Stone'
df.loc[df['label'].str.contains('Normal', case=False, na=False), 'label'] = 'No Stone'


df.to_csv(dataset_path, index=False)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input_shape = (64, 64, 1)
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='sigmoid')
])

model.summary()

import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

df = pd.read_csv('z.csv')


X = np.array(df.iloc[:, :-1])

y = df.iloc[:, -1].apply(lambda x: 0 if 'No Stone' in x else 1)


X = X.reshape(-1, 64, 64, 1)

y = keras.utils.to_categorical(y, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

input_shape = (64, 64, 1)

model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.save("my_cnn_model.h5")

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Testing loss: {loss}, Testing accuracy: {accuracy}')

import cv2

import numpy as np

model = tf.keras.models.load_model("my_cnn_model.h5")

image = cv2.imread('/content/gdrive/MyDrive/Stone/Stone- (10).jpg')


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blur = cv2.GaussianBlur(gray, (5, 5), 0)


_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


largest_contour = max(contours, key=cv2.contourArea)


mask = np.zeros_like(gray)


cv2.drawContours(mask, [largest_contour], -1, (255), cv2.FILLED)


segmented_image = cv2.bitwise_and(image, image, mask=mask)

from google.colab.patches import cv2_imshow
cv2_imshow(segmented_image)

import cv2
import os
import numpy as np

gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(binary, kernel, iterations = 1)
dilation = cv2.dilate(erosion, kernel, iterations = 1)

num_stones = len(cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
stone_size = np.sum(dilation) / 255

contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


marked_image = segmented_image.copy()
min_area_threshold=100
for contour in contours:

    area = cv2.contourArea(contour)


    if area < min_area_threshold :

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


num_rectangles = 0


marked_image = segmented_image.copy()
min_area_threshold = 100
for contour in contours:



    if area < min_area_threshold:

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)


        num_rectangles += 1


if num_rectangles > 0:

    print("Number of rectangles marked: {}".format(num_rectangles))
else:
    print("No stones detected")


from google.colab.patches import cv2_imshow
cv2_imshow(marked_image)

stone_sizes = []
for contour in contours:
    area = cv2.contourArea(contour)

    if area > min_area_threshold:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)


        stone_size = (w, h)
        stone_sizes.append(stone_size)


for i, size in enumerate(stone_sizes):
    if size[0] > 50 and size[1] > 50:
        print("Stone: Width = {}, Height = {}".format( size[0], size[1]))

stone_diameters = []

for contour in contours:
    area = cv2.contourArea(contour)

    if area > min_area_threshold:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 255, 0), 2)


        stone_diameter = max(w, h)
        stone_diameters.append(stone_diameter)


for i, diameter in enumerate(stone_diameters):
    size = stone_sizes[i]
    if size[0] > 50 and size[1] > 50 and size[1] > 50:
        print(" Stone : Diameter = {}mm".format(diameter))

from google.colab.patches import cv2_imshow


contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


marked_image = segmented_image.copy()
min_area_threshold = 100.0
stone_sizes = []

for contour in contours:
    area = cv2.contourArea(contour)

    if area > min_area_threshold:
      x, y, w, h = cv2.boundingRect(contour)



      stone_size = (w, h)
      stone_sizes.append(stone_size)


for i, size in enumerate(stone_sizes):
    if size[0] >50 and size[1] > 50:
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("Stone Deteced")
        print("Stone: Width = {}, Height = {}".format(size[0], size[1]))


cv2_imshow(marked_image)
