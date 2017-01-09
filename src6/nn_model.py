from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model, model_from_yaml
from keras.layers import Dense, Dropout, Flatten, merge, Input, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping
import os
from sklearn.metrics import confusion_matrix, classification_report


class TripletModel:
    def __init__(self, deep_id_dim=3000, aux_weight=0.1, nb_epoch=20, nb_classes=5043, model_name='my_model1'):
        self.batch_size = 100
        self.nb_epoch = nb_epoch
        self.vision_model = Sequential()
        self.model = None
        self.hash_len = deep_id_dim
        self.aux_weight = aux_weight
        self.nb_classes = nb_classes
        self.model_name = model_name
        self.build_model2()

    def build_model2(self):
        def euclidean_distance(vecs):
            x, y = vecs
            return K.sum(K.square(x - y), axis=1, keepdims=True)

        def euclidean_dist_output_shape(shapes):
            shape1, _ = shapes
            return shape1[0], 1

        def triplet_loss(y_true, y_pred):
            # Use y_true as alpha
            mse0, mse1 = y_pred[:, 0], y_pred[:, 1]
            return K.maximum(0.0, mse0 - mse1 + y_true[:, 0])

        # input image dimensions
        img_rows, img_cols, img_channel = 100, 100, 3
        # number of convolutional filters to use
        nb_filters = 20
        # size of pooling area for max pooling
        nb_pool = 2
        # convolution kernel size
        nb_conv = 5

        # build a vision model
        self.vision_model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu',
                                            input_shape=(img_channel, img_rows, img_cols)))
        self.vision_model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
        self.vision_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        self.vision_model.add(Dropout(0.25))
        self.vision_model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
        self.vision_model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
        self.vision_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        self.vision_model.add(Flatten())
        self.vision_model.add(Dense(self.hash_len))  # TODO: tunable!

        img1 = Input(shape=(img_channel, img_rows, img_cols), name='X1')
        img2 = Input(shape=(img_channel, img_rows, img_cols), name='X2')
        img3 = Input(shape=(img_channel, img_rows, img_cols), name='X3')
        hash1, hash2 = self.vision_model(img1), self.vision_model(img2)
        hash3 = self.vision_model(img3)
        vid = Dense(self.nb_classes, activation='softmax', name='aux_output')(hash1)

        distance_layer = Lambda(euclidean_distance, output_shape=euclidean_dist_output_shape)
        dist12 = distance_layer([hash1, hash2])
        dist13 = distance_layer([hash1, hash3])
        merged_out = merge([dist12, dist13], mode='concat', name='main_output')
        self.model = Model(input=[img1, img2, img3], output=[merged_out, vid])
        self.model.summary()
        print(self.model.output_shape)
        print('DeepID dim:', self.hash_len)
        self.model.compile(optimizer='adadelta',
                           loss={'main_output': triplet_loss, 'aux_output': 'categorical_crossentropy'},
                           loss_weights={'main_output': 1., 'aux_output': self.aux_weight})

    def fit(self, X_trains, y_train):
        X_train1, X_train2, X_train3 = X_trains
        main_target, X1_vid = y_train
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        print(X_train1.shape)
        print(X1_vid.shape)
        print(main_target.shape)
        self.model.fit({'X1': X_train1, 'X2': X_train2, 'X3': X_train3},
                       {'main_output': main_target, 'aux_output': X1_vid},
                       batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=1,
                       validation_data=([X_train1, X_train2, X_train3], y_train), callbacks=[early_stopping])
        y_target = np.argmax(X1_vid, axis=1)
        y_predict = np.argmax(self.vision_model.predict(X_train1, verbose=0), axis=1)
        conf_mat = confusion_matrix(y_target, y_predict)
        print('Test accuracy:')
        n_correct = np.sum(np.diag(conf_mat))
        print('# correct:', n_correct, 'out of', len(y_target), ', acc=', float(n_correct) / len(y_target))

    def save_model(self, overwrite=True):
        model_path = '../model/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        # save the wrapper distance model
        yaml_string = self.model.to_yaml()
        open(os.path.join(model_path, self.model_name + '_arch.yaml'), 'w').write(yaml_string)
        self.model.save_weights(os.path.join(model_path, self.model_name + '_weights.h5'), overwrite)

        # save the inner vision model
        model_name = self.model_name + '_vision'
        yaml_string = self.vision_model.to_yaml()
        open(os.path.join(model_path, model_name + '_arch.yaml'), 'w').write(yaml_string)
        self.vision_model.save_weights(os.path.join(model_path, model_name + '_weights.h5'), overwrite)

    def load_model(self):
        model_path = '../model/'
        self.model = model_from_yaml(open(os.path.join(model_path, self.model_name + '_arch.yaml')).read())
        self.model.load_weights(os.path.join(model_path, self.model_name + '_weights.h5'))

        model_name = self.model_name + '_vision'
        self.vision_model = model_from_yaml(open(os.path.join(model_path, model_name + '_arch.yaml')).read())
        self.vision_model.load_weights(os.path.join(model_path, model_name + '_weights.h5'))

    def get_deep_id(self, cars):
        return self.vision_model.predict(cars)
