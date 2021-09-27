import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

class DTClassifier():
    def decisionTreeClassifier(self, x_train, x_test, y_train):
        t0 = time.time()
        dt_clf = DecisionTreeClassifier\
                 (criterion='gini', ccp_alpha=0.0021)\
                 .fit(x_train, y_train)
        t1 = time.time()
        y_predict_train = dt_clf.predict(x_train)
        y_predict_test = dt_clf.predict(x_test)
        #print(learning_curve(dt_clf, x_train, y_train, cv=10, scoring='accuracy'))
        return y_predict_train, y_predict_test, t1 - t0
    
    def dtTrainAccuracy(self, y_train, y_predict_train):
        train_accuracy = accuracy_score(y_train, y_predict_train)
        return train_accuracy

    def dtTestAccuracy(self, y_test, y_predict_test):
        test_accuracy = accuracy_score(y_test, y_predict_test)
        return test_accuracy

    def pruning(self, x_train, y_train):
        dt_clf = DecisionTreeClassifier()
        path = dt_clf.cost_complexity_pruning_path(x_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        return ccp_alphas, impurities

    def hyperParameterTuning(self, x_train, y_train, ccp_alphas, impurities):
        dt_clf = DecisionTreeClassifier()
        params = {'criterion': ['gini', 'entropy'], 'ccp_alpha': ccp_alphas}
        gscv_dt = GridSearchCV(dt_clf, params, cv=10)
        gscv_dt.fit(x_train, y_train)
        return gscv_dt

class NNModel():
    def mlpClassifier(self, x_train, x_test, y_train):
        y_train = y_train.values.ravel()
        t0 = time.time()
        mlp_clf = MLPClassifier(hidden_layer_sizes=61, activation='logistic')\
                  .fit(x_train, y_train)
        t1 = time.time()
        y_predict_train = mlp_clf.predict(x_train)
        y_predict_test = mlp_clf.predict(x_test)
        #print(learning_curve(mlp_clf, x_train, y_train, cv=10, scoring='accuracy'))
        return y_predict_train, y_predict_test, t1 - t0

    def mlpTrainAccuracy(self, y_train, y_predict_train):
        train_accuracy = accuracy_score(y_train, y_predict_train)
        return train_accuracy

    def mlpTestAccuracy(self, y_test, y_predict_test):
        test_accuracy = accuracy_score(y_test, y_predict_test)
        return test_accuracy

    def hyperParameterTuning(self, x_train, y_train):
        y_train = y_train.values.ravel()
        mlp_clf = MLPClassifier()
        params = {'activation': ['identity', 'logistic', 'tanh', 'relu'],\
                  'hidden_layer_sizes': list(range(50, 150))}
        gscv_mlp = GridSearchCV(mlp_clf, params, cv=10)
        gscv_mlp.fit(x_train, y_train)
        return gscv_mlp

class GradientBoost():
    def gradientBoostClassifier(self, x_train, x_test, y_train):
        y_train = y_train.values.ravel()
        t0 = time.time()
        gradientBoost_clf = GradientBoostingClassifier\
                            (criterion='friedman_mse', ccp_alpha=0.0)\
                            .fit(x_train, y_train)
        t1 = time.time()
        y_predict_train = gradientBoost_clf.predict(x_train)
        y_predict_test = gradientBoost_clf.predict(x_test)
        #print(learning_curve(gradientBoost_clf, x_train, y_train, cv=10, scoring='accuracy'))
        return y_predict_train, y_predict_test, t1 - t0
    
    def gradientBoostTrainAccuracy(self, y_train, y_predict_train):
        train_accuracy = accuracy_score(y_train, y_predict_train)
        return train_accuracy
    
    def gradientBoostTestAccuracy(self, y_test, y_predict_test):
        test_accuracy = accuracy_score(y_test, y_predict_test)
        return test_accuracy

    def hyperParameterTuning(self, x_train, y_train, ccp_alphas, impurities):
        y_train = y_train.values.ravel()
        gradientBoost_clf = GradientBoostingClassifier()
        params = {'ccp_alpha': ccp_alphas, 'criterion': ['friedman_mse', 'mse', 'mae']}
        gscv_dt = GridSearchCV(gradientBoost_clf, params, cv=10)
        gscv_dt.fit(x_train, y_train)
        return gscv_dt

class SupportVectorMachine():
    def dataPreProcess(self, x_train, x_test):
        scaler = StandardScaler()
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        return scaled_x_train, scaled_x_test
    
    def svcClassifier(self, scaled_x_train, scaled_x_test, y_train):
        y_train = y_train.values.ravel()
        t0 = time.time()
        svc_clf = SVC(kernel='linear', gamma='scale').fit(scaled_x_train, y_train)
        t1 = time.time()
        y_predict_train = svc_clf.predict(scaled_x_train)
        y_predict_test = svc_clf.predict(scaled_x_test)
        #print(learning_curve(svc_clf, scaled_x_train, y_train, cv=10, scoring='accuracy'))
        return y_predict_train, y_predict_test, t1 - t0

    def svcTrainAccuracy(self, y_train, y_predict_train):
        train_accuracy = accuracy_score(y_train, y_predict_train)
        return train_accuracy

    def svcTestAccuracy(self, y_test, y_predict_test):
        test_accuracy = accuracy_score(y_test, y_predict_test)
        return test_accuracy

    def hyperParameterTuning(self, x_train, y_train):
        y_train = y_train.values.ravel()
        svc_clf = SVC()
        params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],\
                  'gamma': ['scale', 'auto']}
        gscv_mlp = GridSearchCV(svc_clf, params, cv=10)
        gscv_mlp.fit(x_train, y_train)
        return gscv_mlp

class KNN():
    def knnClassifier(self, x_train, x_test, y_train):
        y_train = y_train.values.ravel()
        t0 = time.time()
        knn_clf = KNeighborsClassifier(n_neighbors=16).fit(x_train, y_train)
        t1 = time.time()
        y_predict_train = knn_clf.predict(x_train)
        y_predict_test = knn_clf.predict(x_test)
        #print(learning_curve(knn_clf, x_train, y_train, cv=10, scoring='accuracy'))
        return y_predict_train, y_predict_test, t1 - t0
    
    def knnTrainAccuracy(self, y_train, y_predict_train):
        train_accuracy = accuracy_score(y_train, y_predict_train)
        return train_accuracy
    
    def knnTestAccuracy(self, y_test, y_predict_test):
        test_accuracy = accuracy_score(y_test, y_predict_test)
        return test_accuracy

    def hyperParameterTuning(self, x_train, y_train):
        y_train = y_train.values.ravel()
        knn_clf = KNeighborsClassifier()
        params = {'n_neighbors': list(range(1, 30)),\
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        gscv_mlp = GridSearchCV(knn_clf, params, cv=10)
        gscv_mlp.fit(x_train, y_train)
        return gscv_mlp
        
def main():
    nasa_train_accuracy = np.zeros(5)
    nasa_test_accuracy = np.zeros(5)

    water_train_accuracy = np.zeros(5)
    water_test_accuracy = np.zeros(5)

    nasa_train_time = np.zeros(5)

    water_train_time = np.zeros(5)
    
    nasa_data = 'nasa.csv'
    x_nasa, y_nasa = Data().dataAllocation(nasa_data)
    del x_nasa['Close Approach Date']
    del x_nasa['Orbiting Body']
    del x_nasa['Orbit Determination Date']
    del x_nasa['Equinox']
    y_nasa['Hazardous'] = y_nasa['Hazardous'].astype(int)

    #
    x_nasa_train, x_nasa_test, y_nasa_train, y_nasa_test = Data().trainSets(x_nasa, y_nasa)

    water_data = 'water_potability.csv'
    x_water, y_water = Data().dataAllocation(water_data)

    #
    x_water_train, x_water_test, y_water_train, y_water_test = Data().trainSets(x_water, y_water)

    y_nasa_dtpredict_train, y_nasa_dtpredict_test, nasa_train_time[0] = DTClassifier().decisionTreeClassifier\
                                                (x_nasa_train, x_nasa_test, y_nasa_train)

    nasa_train_accuracy[0] = DTClassifier().dtTrainAccuracy(y_nasa_train, y_nasa_dtpredict_train)
    nasa_test_accuracy[0] = DTClassifier().dtTestAccuracy(y_nasa_test, y_nasa_dtpredict_test)

    y_water_dtpredict_train, y_water_dtpredict_test, water_train_time[0] = DTClassifier().decisionTreeClassifier\
                                                (x_water_train, x_water_test, y_water_train)
    
    water_train_accuracy[0] = DTClassifier().dtTrainAccuracy(y_water_train, y_water_dtpredict_train)
    water_test_accuracy[0] = DTClassifier().dtTestAccuracy(y_water_test, y_water_dtpredict_test)

    y_nasa_mlppredict_train, y_nasa_mlppredict_test, nasa_train_time[1] = NNModel().mlpClassifier\
                                                (x_nasa_train, x_nasa_test, y_nasa_train)

    nasa_train_accuracy[1] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train)
    nasa_test_accuracy[1] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test)

    y_water_mlppredict_train, y_water_mlppredict_test, water_train_time[1] = NNModel().mlpClassifier\
                                                (x_water_train, x_water_test, y_water_train)

    water_train_accuracy[1] = NNModel().mlpTrainAccuracy(y_water_train, y_water_mlppredict_train)
    water_test_accuracy[1] = NNModel().mlpTestAccuracy(y_water_test, y_water_mlppredict_test)

    y_nasa_gradientBoost_train, y_nasa_gradientBoost_test, nasa_train_time[2] = \
                                GradientBoost().gradientBoostClassifier\
                                (x_nasa_train, x_nasa_test, y_nasa_train)
    
    nasa_train_accuracy[2] = GradientBoost().gradientBoostTrainAccuracy(y_nasa_train,\
                                                    y_nasa_gradientBoost_train)
    nasa_test_accuracy[2] = GradientBoost().gradientBoostTestAccuracy(y_nasa_test,\
                                                    y_nasa_gradientBoost_test)

    y_water_gradientBoost_train, y_water_gradientBoost_test, water_train_time[2] = \
                                 GradientBoost().gradientBoostClassifier\
                                 (x_water_train, x_water_test, y_water_train)
    
    water_train_accuracy[2] = GradientBoost().gradientBoostTrainAccuracy(y_water_train,\
                                                     y_water_gradientBoost_train)
    water_test_accuracy[2] = GradientBoost().gradientBoostTestAccuracy(y_water_test,\
                                                    y_water_gradientBoost_test)

    scaled_x_nasa_train, scaled_x_nasa_test = SupportVectorMachine()\
                                              .dataPreProcess(x_nasa_train, x_nasa_test)    
    y_nasa_svcpredict_train, y_nasa_svcpredict_test, nasa_train_time[3] = SupportVectorMachine().svcClassifier\
                                                (scaled_x_nasa_train, scaled_x_nasa_test, y_nasa_train)
    
    nasa_train_accuracy[3] = SupportVectorMachine().svcTrainAccuracy(y_nasa_train, y_nasa_svcpredict_train)
    nasa_test_accuracy[3] = SupportVectorMachine().svcTestAccuracy(y_nasa_test, y_nasa_svcpredict_test)

    scaled_x_water_train, scaled_x_water_test = SupportVectorMachine()\
                                              .dataPreProcess(x_water_train, x_water_test)
    y_water_svcpredict_train, y_water_svcpredict_test, water_train_time[3] = SupportVectorMachine().svcClassifier\
                                                (scaled_x_water_train, scaled_x_water_test,\
                                                 y_water_train)
    
    water_train_accuracy[3] = SupportVectorMachine().svcTrainAccuracy(y_water_train, y_water_svcpredict_train)
    water_test_accuracy[3] = SupportVectorMachine().svcTestAccuracy(y_water_test, y_water_svcpredict_test)

    y_nasa_knn_train, y_nasa_knn_test, nasa_train_time[4] = KNN().knnClassifier\
                                        (x_nasa_train, x_nasa_test, y_nasa_train)
    
    nasa_train_accuracy[4] = KNN().knnTrainAccuracy(y_nasa_train, y_nasa_knn_train)
    nasa_test_accuracy[4] = KNN().knnTestAccuracy(y_nasa_test, y_nasa_knn_test)

    y_water_knn_train, y_water_knn_test, water_train_time[4] = KNN().knnClassifier\
                                          (x_water_train, x_water_test, y_water_train)
    
    water_train_accuracy[4] = KNN().knnTrainAccuracy(y_water_train, y_water_knn_train)
    water_test_accuracy[4] = KNN().knnTestAccuracy(y_water_test, y_water_knn_test)

    a_nasa_dt, b_nasa_dt = DTClassifier().pruning(x_nasa_train, y_nasa_train)
    a_water_dt, b_water_dt = DTClassifier().pruning(x_water_train, y_water_train)

    '''print(DTClassifier().hyperParameterTuning\
          (x_nasa_train, y_nasa_train, a_nasa_dt, b_nasa_dt).best_params_)
    print(DTClassifier().hyperParameterTuning\
          (x_water_train, y_water_train, a_water_dt, b_water_dt).best_params_, '\n')

    print(NNModel().hyperParameterTuning\
          (x_nasa_train, y_nasa_train).best_params_)
    print(NNModel().hyperParameterTuning\
          (x_water_train, y_water_train).best_params_, '\n')'''
    
    '''print(GradientBoost().hyperParameterTuning\
          (x_nasa_train, y_nasa_train, a_nasa_boost, b_nasa_boost).best_params_)
    print(GradientBoost().hyperParameterTuning\
          (x_water_train, y_water_train, a_water_boost, b_water_boost).best_params_, '\n')'''
    
    '''

    print(validation_curve(DecisionTreeClassifier(), x_nasa_train, y_nasa_train.values.ravel(),\
                           'criterion', ['gini', 'entropy'], cv=10, scoring='accuracy'))
    print(validation_curve(DecisionTreeClassifier(), x_water_train, y_water_train.values.ravel(),\
                           'criterion', ['gini', 'entropy'], cv=10, scoring='accuracy'))
                           
    print(validation_curve(DecisionTreeClassifier(), x_nasa_train, y_nasa_train,\
                           'ccp_alpha', a_nasa_dt, cv=10, scoring='accuracy'))
    print(validation_curve(DecisionTreeClassifier(), x_water_train, y_water_train,\
                           'ccp_alpha', a_water_dt, cv=10, scoring='accuracy'))
                           
        
    print(validation_curve(MLPClassifier(), x_nasa_train, y_nasa_train.values.ravel(),\
                           'activation', ['identity', 'logistic', 'tanh', 'relu'],\
                           cv=10, scoring='accuracy'))
    print(validation_curve(MLPClassifier(), x_water_train, y_water_train.values.ravel(),\
                           'activation', ['identity', 'logistic', 'tanh', 'relu'],\
                           cv=10, scoring='accuracy'))

    print(validation_curve(MLPClassifier(), x_nasa_train, y_nasa_train.values.ravel(),\
                           'hidden_layer_sizes', list(range(50, 150)),\
                           cv=10, scoring='accuracy'))
    print(validation_curve(MLPClassifier(), x_water_train, y_water_train.values.ravel(),\
                           'hidden_layer_sizes', list(range(50, 150)),\
                           cv=10, scoring='accuracy'))
    
    print(validation_curve(GradientBoostingClassifier(), x_nasa_train, y_nasa_train.values.ravel(),\
                           'ccp_alpha', a_nasa_dt, cv=10, scoring='accuracy'))
    print(validation_curve(GradientBoostingClassifier(), x_water_train, y_water_train.values.ravel(),\
                           'ccp_alpha', a_water_dt, cv=10, scoring='accuracy'))
    
    print(validation_curve(GradientBoostingClassifier(), x_nasa_train, y_nasa_train.values.ravel(),\
                           'criterion', ['friedman_mse', 'mse', 'mae'], cv=10, scoring='accuracy'))
    print(validation_curve(GradientBoostingClassifier(), x_water_train, y_water_train.values.ravel(),\
                           'criterion', ['friedman_mse', 'mse', 'mae'], cv=10, scoring='accuracy'))

    print(validation_curve(SVC(), scaled_x_nasa_train, y_nasa_train.values.ravel(),\
                           'kernel', ['linear', 'poly', 'rbf', 'sigmoid'], cv=10,\
                           scoring='accuracy'))
    print(validation_curve(SVC(), scaled_x_water_train, y_water_train.values.ravel(),\
                           'kernel', ['linear', 'poly', 'rbf', 'sigmoid'], cv=10,\
                           scoring='accuracy'))
                           
    print(validation_curve(SVC(), scaled_x_nasa_train, y_nasa_train.values.ravel(),\
                           'gamma', ['scale', 'auto'], cv=10,\
                           scoring='accuracy'))
    print(validation_curve(SVC(), scaled_x_water_train, y_water_train.values.ravel(),\
                           'gamma', ['scale', 'auto'], cv=10,\
                           scoring='accuracy'))
    
    print(validation_curve(KNeighborsClassifier(), x_nasa_train, y_nasa_train.values.ravel(),\
                           'n_neighbors', list(range(1, 30)), cv=10, scoring='accuracy'))
    print(validation_curve(KNeighborsClassifier(), x_water_train, y_water_train.values.ravel(),\
                           'n_neighbors', list(range(1, 30)), cv=10, scoring='accuracy'))

    print(validation_curve(KNeighborsClassifier(), x_nasa_train, y_nasa_train.values.ravel(),\
                           'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'],\
                           cv=10, scoring='accuracy'))
    print(validation_curve(KNeighborsClassifier(), x_water_train, y_water_train.values.ravel(),\
                           'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'],\
                           cv=10, scoring='accuracy'))'''


    classifiers = ('Decision Tree', 'Neural Network', 'Boosting', 'SVM', 'KNN')
    y_pos = np.arange(len(classifiers))

    plt.figure()
    plt.barh(y_pos, nasa_test_accuracy)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_xlim(0.8, 1.0)
    plt.gca().set_yticklabels(classifiers)
    plt.gca().invert_yaxis()  # labels read top-to-bottom
    plt.title('Comparison of Accuracies')
    plt.xlabel('Accuracy')
    plt.subplots_adjust(left=0.15)
    plt.show()
    
    '''plt.figure()
    plt.barh(y_pos, nasa_train_time)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_yticklabels(classifiers)
    plt.gca().invert_yaxis()  # labels read top-to-bottom
    plt.title('Comparison of Training Time')
    plt.xlabel('Training Time (in seconds)')
    plt.subplots_adjust(left=0.15)
    plt.show()'''
        
if __name__ == "__main__":
	main()
