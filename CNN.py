from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from DataLoader import DataLoader
import Constants as Constants


class NeuralNetworkModel:
    def __init__(self):
        self.dataLoader = DataLoader()
        self.model = None

    def create_model(self, learning_rate=0.03, learning_decay=1e-5, learning_momentum=0.4):
        #reference https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        inputs = Input(shape=(Constants.FACE_SIZE, Constants.FACE_SIZE, 1))
        x = Conv2D(filters=64, kernel_size=5, activation='relu', input_shape=(Constants.FACE_SIZE, Constants.FACE_SIZE, 1))(inputs)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Conv2D(filters=64, kernel_size=5, activation='relu')(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Conv2D(filters=128, kernel_size=4, activation='relu')(x)

        # avoid overfitting
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        outputs = Dense(units=len(Constants.EMOTIONS), activation='softmax')(x)

        self.model = Model(inputs, outputs)
        # Stochastic gradient descent (SGD)
        sgd = SGD(lr=learning_rate, decay=learning_decay, momentum=learning_momentum)
        self.model.compile(loss='mse', optimizer=sgd)  # mean squared error

    def train_model(self, training_epochs=200, training_batch_size=50):
        x_train, x_test, y_train, y_test = self.dataLoader.load_data()
        print('Training model...')
        self.model.fit(x=x_train, y=y_train, epochs=training_epochs, batch_size=training_batch_size, verbose=1, shuffle=True)

    def evaluate_model(self, eval_batch_size=50):
        x_train, x_test, y_train, y_test = self.dataLoader.load_data()
        print('Evaluating model...')
        evaluation = self.model.evaluate(x_test, y_test, batch_size=eval_batch_size, verbose=1)
        return evaluation

    def predict(self, image):
        if image is not None:
            image = image.reshape([-1, Constants.FACE_SIZE, Constants.FACE_SIZE, 1])
            return self.model.predict(image)
        else:
            return None
