import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import models, datasets
from tensorflow.keras.applications import VGG16

# Denoising autoencoder model
def ae_model():
    model = Sequential()
    model.add(layers.Input(shape=(32, 32, 3))) # cifar10 data shape (32, 32, 3)

    # Encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Decoder
    model.add(Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"))
    model.add(Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding="same"))

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model

# dataset 전처리
def preprocess(array):

    array = array.astype("float32") / 255.
    array = np.reshape(array, (len(array), 32, 32, 3))

    return array

# Noise filter
def noise(array):

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=array.shape)

    '''
    numpy.clip(array, min, max)
    array 내의 element들에 대해서
    min 값 보다 작은 값들을 min 값
    max 값 보다 큰 값들을 max 값
    '''
    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2, array3):

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    images3 = array3[indices, :]

    plt.figure(figsize=(20, 4))

    for i, (image1, image2, image3) in enumerate(zip(images1, images2, images3)):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(image1.reshape(32, 32, 3)) # (28, 28)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(image2.reshape(32, 32, 3)) # (28, 28)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n*2)
        plt.imshow(image3.reshape(32, 32, 3))  # (28, 28)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

# 하이퍼 파라메터
MY_EPOCH = 100
MY_BATCHSIZE = 256
NAME = "cifar10"
filename = f"{NAME}_e{MY_EPOCH}_b{MY_BATCHSIZE}.h5"

def train_noisy(model, noisy_train_data, train_data, noisy_test_data, test_data):
    history = model.fit(x=noisy_train_data, y=train_data, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE, shuffle=True, validation_data=(noisy_test_data, test_data))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'./{NAME}_accuracy_EPOCH{MY_EPOCH}_BATCH{MY_BATCHSIZE}.png')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'./{NAME}_loss_EPOCH{MY_EPOCH}_BATCH{MY_BATCHSIZE}.png')
    plt.show()

    model.save(filename)

def test(test_data, noisy_test_data):
    model = load_model(filename)

    predictions = model.predict(noisy_test_data)

    display(test_data, noisy_test_data, predictions)

    for i in range(predictions.shape[0]):

        cv.namedWindow("Resized test_data", cv.WINDOW_NORMAL)
        cv.namedWindow("Resized prediction", cv.WINDOW_NORMAL)
        cv.namedWindow("Resized cha", cv.WINDOW_NORMAL)
        cv.resizeWindow('Resized test_data', 320, 320)
        cv.resizeWindow('Resized prediction', 320, 320)
        cv.resizeWindow('Resized cha', 320, 320)

        tmp = test_data[i] - predictions[i]

        print(tmp, type(tmp))

        cv.imshow("Resized cha", tmp)
        cv.imshow("Resized test_data", test_data[i])
        cv.imshow("Resized prediction", predictions[i])
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == "__main__":

    # cifar10 dataset load
    cifar10 = datasets.cifar10
    (train_data, _), (test_data, _) = cifar10.load_data()

    # dataset을 255. 로 나누어 정규화 (0~1사이의 값에서 최적의 성능을 냄)
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    # noise filter
    noisy_train_data = noise(train_data)
    noisy_test_data = noise(test_data)

    # display(train_data, noisy_train_data)

    '''
    noisy, normal 학습
    '''
    # 학습
    noise_factor : 0.4
    my_model = ae_model()
    train_noisy(my_model, noisy_train_data, train_data, noisy_test_data, test_data)

    #테스트
    test(test_data, noisy_test_data)