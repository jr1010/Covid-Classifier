import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import model_from_json
import pickle


def diagnosis(file):
    # Download Image
    try:
        image = cv2.imread(file)
    except:
        print("Cannot Download image: ,", file)
        return

    # Image Processing
    IMM_SIZE = 224
    image = cv2.resize(image, (IMM_SIZE, IMM_SIZE))

    if len(image.shape) > 2:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    # Display Image
    plt.gray()
    plt.imshow(image)
    plt.show()

    # Load model

    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    # Loading weights
    model.load_weights('model.h5')

    with open('lab.pickle', 'rb') as f:
        lab = pickle.load(f)

    # Normalize the image data

    image = np.array(image) / 255

    ##Reshape the image

    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

    # Prediction

    diag = np.argmax(model.predict(image))

    diag = list(lab.keys())[list(lab.values()).index(diag)]

    return diag

print("Diagnosis :",diagnosis(r"C:\Users\Jai\PycharmProjects\COVID_Classifier\020.jpg"))