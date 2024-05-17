!pip install opencv-python
!pip install keras
!pip install tensorflow
# from google.colab import drive
# drive.mount('/content/drive')
import numpy as np
#from google.colab.patches import cv2_imshow
import cv2
import os
import pandas as pd


import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt

ruta_directorio = "train"
ruta_csv = "_classes2.csv"

# Leer el archivo CSV utilizando pandas
df = pd.read_csv(ruta_csv)

# Diccionario para almacenar los valores de la segunda y tercera columna del CSV
valores_csv = {}
# Llenar el diccionario con los valores del CSV
for index, row in df.iterrows():
    nombre_imagen = row['filename'].split('.')[0]  # Eliminar la extensión para comparar con la imagen
    valores_csv[nombre_imagen] = (row['blue_ball'], row['purple_ball'], row['red_ball'])

# Lista para almacenar las imágenes y sus etiquetas
imagenes = []
etiquetas1=[]
etiquetas = {}

# Iterar sobre los archivos en el directorio
for archivo in os.listdir(ruta_directorio):
    # Verificar si el archivo es una imagen JPG
    if archivo.endswith(".jpg"):
        #print(valores_csv[archivo.split('.')[0]])

        # Leer la imagen utilizando OpenCV
        ruta_imagen = os.path.join(ruta_directorio, archivo)
        imagen = cv2.imread(ruta_imagen)
        # Agregar la imagen a la lista
        if imagen is not None:
            imagenes.append(imagen)
            # Obtener la etiqueta correspondiente del archivo CSV
            nombre_imagen = archivo.split('.')[0]  # Eliminar la extensión para buscar en el diccionario
            if nombre_imagen in valores_csv:
                etiquetas1.append(valores_csv[archivo.split('.')[0]])
                etiquetas[archivo.split('.')[0]]=(valores_csv[archivo.split('.')[0]])
            else:
                print(f"No se encontraron etiquetas para la imagen: {nombre_imagen}")
        else:
            print(f"No se pudo leer la imagen: {ruta_imagen}")

# Ahora puedes acceder a tus imágenes y etiquetas correspondientes
# Por ejemplo, para mostrar la primera imagen y su etiqueta:
if imagenes:
    plt.imshow(imagenes[0])
    #print("Etiqueta de la imagen:", etiquetas[0])
else:
    print("No se encontraron imágenes en el directorio especificado.")

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def custom_load_data(imgArray, etiquetas, test_size=0.2, target_size=(80, 80)):
  """
  Splits the image array `imgArray` and label array `etiquetas` into training and testing sets.

  Args:
      imgArray (np.ndarray): The NumPy array containing the images.
      etiquetas (np.ndarray): A NumPy array containing the labels for each image.
      test_size (float, optional): The proportion of data to be used for the testing set. Defaults to 0.2.
      target_size (tuple, optional): The target size to resize the images to. Defaults to (80, 80).

  Returns:
      tuple: A tuple containing four elements:
          - x_train (np.ndarray): The training set of images.
          - y_train (np.ndarray): The training set of labels.
          - x_test (np.ndarray): The testing set of images.
          - y_test (np.ndarray): The testing set of labels.
  """

  # Split the data
  x_train, x_test, y_train, y_test = train_test_split(imgArray, etiquetas, test_size=test_size)

  # Resize the images
  x_train_resized = [cv2.resize(img, target_size) for img in x_train]
  x_test_resized = [cv2.resize(img, target_size) for img in x_test]

  # Convert to NumPy arrays
  x_train_resized = np.array(x_train_resized)
  x_test_resized = np.array(x_test_resized)

  # No need to assume a single class anymore, use the actual labels
  y_train = y_train
  y_test = y_test

  return (x_train_resized, y_train), (x_test_resized, y_test)

# Assuming you have loaded your images into a list called 'imagenes'

# Convert the list of images to a NumPy array
imgArray = np.asarray(imagenes)
etiquetas1 = np.asarray(etiquetas1)

# Use the custom function to split data and create labels
(x_train, y_train), (x_test, y_test) = custom_load_data(imgArray, etiquetas1)

# Normalize the data
# x_train = np.divide(x_train.astype('float32'), 255.0)
# x_test = np.divide(x_test.astype('float32'), 255.0)

# Normalize the data
x_train = np.divide(x_train, 255.0)
x_test = np.divide(x_test, 255.0)

# Print the shapes of the arrays
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)


from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters=3,
                 kernel_size=(3,3),
                 strides= (1,1),
                 padding='same',
                 input_shape = (80, 80, 3),
                 activation = 'relu'
                 ))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=3,
                 kernel_size=(3,3),
                 strides= (1,1),
                 padding='same',
                 activation = 'relu'
                 ))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=3,
                 kernel_size=(3,3),
                 strides= (1,1),
                 padding='same',
                 activation = 'relu'
                 ))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=3,
                 kernel_size=(3,3),
                 strides= (1,1),
                 padding='same',
                 activation = 'relu'
                 ))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(filters=3,
                 kernel_size=(3,3),
                 strides= (1,1),
                 padding='same',
                 activation = 'relu'
                 ))

model.add(MaxPooling2D(2,2))




model.add(Flatten())

model.add(Dense(3, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())


import matplotlib.pyplot as plt
import random
num = random.randint(1, 241)
img = x_test[num]
label = y_test[num]

plt.imshow(img)
plt.axis('off')
plt.show()


out = model.predict(img.reshape(1,80,80,3))

print(label)
# print(out)
v = np.round(out).astype(int)
print(v)
# print(y_test)
# print(np.argmax(out))
