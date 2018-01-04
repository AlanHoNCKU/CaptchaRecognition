import os
import numpy as np
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


APPEARED_LETTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z'
]

def load_data(folder):
    img_list = [i for i in os.listdir(folder) if i.endswith('jpg')]
    letters_num = len(img_list) * 5
    print('total letters:', letters_num)
    data = np.empty((letters_num, 40, 40, 3), dtype="uint8")  # channel last
    label = np.empty((letters_num,))
    for index, img_name in enumerate(img_list):
        raw_img = preprocess.load_img(os.path.join(folder, img_name))
        sub_imgs = preprocess.gen_sub_img(raw_img)
        for sub_index, img in enumerate(sub_imgs):
            data[index*5+sub_index, :, :, :] = img / 255
            label[index*5+sub_index] = CHR2CAT[img_name[sub_index]]
        if index % 100 == 0:
            print('{} letters loads'.format(index*5))
    return data, label

def prepare_data(folder):
    print('... loading data')
    letter_num = len(APPEARED_LETTERS)
    data, label = load_data(folder)
    data_train, data_test, label_train, label_test = \
        train_test_split(data, label, test_size=0.1, random_state=0)
    label_categories_train = to_categorical(label_train, letter_num)
    label_categories_test = to_categorical(label_test, letter_num)
    return (data_train, label_categories_train,
            data_test, label_categories_test)


def build_model():
    print('... construct network')
    inputs = layers.Input((40, 40, 3))
    x = layers.Conv2D(32, 9, activation='relu')(inputs)
    x = layers.Conv2D(32, 9, activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(640)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(APPEARED_LETTERS), activation='softmax')(x)

    return Model(inputs=inputs, outputs=out)


def train(pic_folder, weight_folder):
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    x_train, y_train, x_test, y_test = prepare_data(pic_folder)
    model = build_model()

    print('... compile models')
    model.compile(
        optimizer='adadelta',
        loss=['categorical_crossentropy'],
        metrics=['accuracy'],
    )

    print('... begin train')

    check_point = ModelCheckpoint(
        os.path.join(weight_folder, '{epoch:02d}.hdf5'))

    class TestAcc(Callback):
        def on_epoch_end(self, epoch, logs=None):
            weight_file = os.path.join(
                weight_folder, '{epoch:02d}.hdf5'.format(epoch=epoch + 1))
            model.load_weights(weight_file)
            out = model.predict(x_test, verbose=1)
            predict = np.array([np.argmax(i) for i in out])
            answer = np.array([np.argmax(i) for i in y_test])
            acc = np.sum(predict == answer) / len(predict)
            print('Single letter test accuracy: {:.2%}'.format(acc))
            print('Picture accuracy: {:.2%}'.format(np.power(acc, 5)))
            print('----------------------------------\n')

    model.fit(
        x_train, y_train, batch_size=128, epochs=100,
        validation_split=0.1, callbacks=[check_point, TestAcc()],
    )

if __name__ == '__main__':
    train(
        pic_folder=r'/home/chiayu/Documents/Learn_ML/captcha_recognition/TrainData/',
        weight_folder=r'/home/chiayu/Documents/Learn_ML/captcha_recognition/'
    )