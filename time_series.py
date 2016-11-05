"""
Deep learning for speeds prediction.
"""

import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


class DeepLearnSpeeds:
    def __init__(self, num_epoch, look_back, seed):
        self.__num_epoch = num_epoch
        self.__look_back = look_back
        # TODO: make specific to instance
        self.__seed = seed
        numpy.random.seed(seed)

    @staticmethod
    def __load_and_split_dataset(csv_path, nrows=None, train_percent=0.67):
        dataset = pandas \
            .read_csv(csv_path, usecols=[0], engine='python', skipfooter=0, nrows=nrows) \
            .values.astype('float32')
        train_size = int(len(dataset) * train_percent)
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        return dataset, train, test

    # convert an array of values into a dataset matrix
    def __create_dataset_matrix(self, dataset):
        data_x = [dataset[i:(i + self.__look_back), 0] for i in range(len(dataset) - self.__look_back - 1)]
        data_y = [dataset[i + self.__look_back, 0] for i in range(len(dataset) - self.__look_back - 1)]
        return numpy.array(data_x), numpy.array(data_y)

    def __create_model(self):
        # create and fit Multilayer Perceptron model
        model = Sequential()
        model.add(Dense(8, input_dim=self.__look_back, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @staticmethod
    def __add_plot(model, dataset_to_predict_name, full_dataset, dataset_to_predict_x, dataset_to_predict_y,
                   offset_start, offset_end, plt_obj):
        predict = model.predict(dataset_to_predict_x)

        # shift train predictions for plotting
        predict_plot = numpy.empty_like(full_dataset)
        predict_plot[:, :] = numpy.nan
        predict_plot[offset_start:offset_end, :] = predict
        plt_obj.plot(predict_plot)

        errors = predict[:, 0] - dataset_to_predict_y
        max_abs_err = max(enumerate(abs(errors)), key=lambda pair: pair[1])
        print(max_abs_err[0] + offset_start, max_abs_err[1])
        rmse = sum(errors ** 2 / len(errors)) ** 0.5
        avg_abs_err = sum(abs(errors)) / len(errors)
        print('%s dataset: Maximum absolute error is %.2f ; Root of Mean Square Error is %.2f ; average absolute error is %.2f' %
              (dataset_to_predict_name, max_abs_err[1], rmse, avg_abs_err))
        return errors

    def __plot(self, model, dataset, train_x, train_y, test_x, test_y):
        # plot baseline and predictions
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(dataset)
        self.__add_plot(model, 'Train', dataset, train_x, train_y,
                        self.__look_back, len(train_x) + self.__look_back, ax1)
        test_errors = self.__add_plot(model, 'Test', dataset, test_x, test_y,
                                      len(train_x) + (self.__look_back * 2) + 1, len(dataset) - 1, ax1)
        plt.savefig('figures/predictions.svg')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(abs(test_errors))
        plt.savefig('figures/abs_test_errors.svg')

    def predict_and_plot(self, csv_path):
        dataset, train, test = self.__load_and_split_dataset(csv_path)
        train_x, train_y = self.__create_dataset_matrix(train)
        test_x, test_y = self.__create_dataset_matrix(test)
        model = self.__create_model()
        model.fit(train_x, train_y, nb_epoch=self.__num_epoch, batch_size=2, verbose=2)
        self.__plot(model, dataset, train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    deep_learn_speeds = DeepLearnSpeeds(num_epoch=2, look_back=1, seed=7)
    deep_learn_speeds.predict_and_plot('speeds1_h20000_t5000.csv')
