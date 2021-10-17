import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from yellowbrick.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import time

class Data():
    def dataAllocation(self,path):
        df = pd.read_csv(path)
        df = df.dropna(how='any',axis=0)
        x_data = df.iloc[:, :-1]
        y_data = df.iloc[:, -1:]
        return x_data, y_data
    def trainSets(self, x_data, y_data):
        x_train, x_test, y_train, y_test = train_test_split\
                                           (x_data, y_data, train_size = 0.70,\
                                            test_size = 0.30, random_state = 614,\
                                            shuffle = True)
        return x_train, x_test, y_train, y_test

def main():
    nasa_train_accuracy = np.zeros(3)
    nasa_test_accuracy = np.zeros(3)
    nasa_train_time = np.zeros(3)

    nasa_data = 'nasa.csv'
    x_nasa, y_nasa = Data().dataAllocation(nasa_data)
    del x_nasa['Close Approach Date']
    del x_nasa['Orbiting Body']
    del x_nasa['Orbit Determination Date']
    del x_nasa['Equinox']
    y_nasa['Hazardous'] = y_nasa['Hazardous'].astype(int)

    x_nasa_train, x_nasa_test, y_nasa_train, y_nasa_test = \
                  Data().trainSets(x_nasa, y_nasa)

    t0 = time.time()
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [61], activation = 'tanh',\
                                     algorithm = 'random_hill_climb',\
                                     max_iters = 1000, learning_rate = 0.0001,\
                                     clip_max = 5, max_attempts = 100, random_state = 3,\
                                     early_stopping = True)
    nn_model1.fit(x_nasa_train, y_nasa_train)
    t1 = time.time()
    nasa_train_time[0] = t1 - t0
    y_nasa_train_pred1 = nn_model1.predict(x_nasa_train)
    nasa_train_accuracy[0] = accuracy_score(y_nasa_train, y_nasa_train_pred1)
    print(nasa_train_accuracy[0])
    y_nasa_test_pred1 = nn_model1.predict(x_nasa_test)
    nasa_test_accuracy[0] = accuracy_score(y_nasa_test, y_nasa_test_pred1)
    print(nasa_test_accuracy[0])
    print(learning_curve(nn_model1, x_nasa_train, y_nasa_train, scoring='accuracy'))

    t0 = time.time()
    nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [61], activation = 'tanh',\
                                     algorithm = 'simulated_annealing',\
                                     max_iters = 1000, learning_rate = 0.0001,\
                                     clip_max = 5, max_attempts = 100, random_state = 3,\
                                     early_stopping = True)
    nn_model2.fit(x_nasa_train, y_nasa_train)
    t1 = time.time()
    nasa_train_time[1] = t1 - t0
    y_nasa_train_pred2 = nn_model2.predict(x_nasa_train)
    nasa_train_accuracy[1] = accuracy_score(y_nasa_train, y_nasa_train_pred2)
    print(nasa_train_accuracy[1])
    y_nasa_test_pred2 = nn_model2.predict(x_nasa_test)
    nasa_test_accuracy[1] = accuracy_score(y_nasa_test, y_nasa_test_pred2)
    print(nasa_test_accuracy[1])
    #print(learning_curve(nn_model2, x_nasa_train, y_nasa_train, cv=10, scoring='accuracy'))

    t0 = time.time()
    nn_model3 = mlrose.NeuralNetwork(hidden_nodes = [61], activation = 'tanh',\
                                     algorithm = 'genetic_alg',\
                                     max_iters = 1000, learning_rate = 0.0001,\
                                     clip_max = 5, max_attempts = 100, random_state = 3,\
                                     early_stopping = True)
    nn_model3.fit(x_nasa_train, y_nasa_train)
    t1 = time.time()
    nasa_train_time[2] = t1 - t0
    y_nasa_train_pred3 = nn_model3.predict(x_nasa_train)
    nasa_train_accuracy[2] = accuracy_score(y_nasa_train, y_nasa_train_pred3)
    print(nasa_train_accuracy[2])
    y_nasa_test_pred3 = nn_model3.predict(x_nasa_test)
    nasa_test_accuracy[2] = accuracy_score(y_nasa_test, y_nasa_test_pred3)
    print(nasa_test_accuracy[2])
    #print(learning_curve(nn_model3, x_nasa_train, y_nasa_train, cv=10, scoring='accuracy'))

    algorithms = ('Random Hill Climb', 'Simulated Annealing', 'Genetic Algorithm')
    y_pos = np.arange(len(algorithms))

    plt.figure()
    plt.barh(y_pos, nasa_train_accuracy)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_xlim(0.8, 0.9)
    plt.gca().set_yticklabels(algorithms)
    plt.gca().invert_yaxis()
    plt.title('Comparison of Accuracies')
    plt.xlabel('Accuracy')
    plt.subplots_adjust(left=0.2)
    plt.show()

    '''plt.figure()
    plt.barh(y_pos, nasa_train_time)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_yticklabels(algorithms)
    plt.gca().invert_yaxis()
    plt.title('Comparison of Training Time')
    plt.xlabel('Training Time (in seconds)')
    plt.subplots_adjust(left=0.2)
    plt.show()'''
    
if __name__ == "__main__":
	main()
