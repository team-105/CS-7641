import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.neural_network import MLPClassifier
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.features import ParallelCoordinates
from yellowbrick.features.pcoords import parallel_coordinates
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

class NNModel():
    def mlpClassifier(self, x_train, x_test, y_train):
        y_train = y_train.values.ravel()
        t0 = time.time()
        mlp_clf = MLPClassifier(hidden_layer_sizes=84, activation='logistic', random_state=0)\
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
        
def main():
    nasa_train_accuracy = np.zeros(5)
    nasa_test_accuracy = np.zeros(5)

    nasa_train_time = np.zeros(5)
    
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

    '''nasa_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    nasa_visualizer.fit(x_nasa_train, y_nasa_train)
    nasa_visualizer.show()

    kmeans_nasa = KMeans(n_clusters=6, random_state=0).fit(x_nasa_train, y_nasa_train)
    samples_nasa = x_nasa_train.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = kmeans_nasa.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''kmeans_nasa = KMeans(n_clusters=6, random_state=0).fit(x_nasa_train, y_nasa_train)
    x_nasa_train_kmeans = kmeans_nasa.transform(x_nasa_train)
    x_nasa_test_kmeans = kmeans_nasa.transform(x_nasa_test)

    y_nasa_mlppredict_train, y_nasa_mlppredict_test, nasa_train_time[0] = \
                             NNModel().mlpClassifier(x_nasa_train, x_nasa_test, y_nasa_train)
    
    y_nasa_mlppredict_train_kmeans, y_nasa_mlppredict_test_kmeans, nasa_train_time[1] = \
                             NNModel().mlpClassifier(x_nasa_train_kmeans, x_nasa_test_kmeans, \
                                                     y_nasa_train)

    nasa_train_accuracy[0] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train)
    nasa_test_accuracy[0] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test)

    nasa_train_accuracy[1] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train_kmeans)
    nasa_test_accuracy[1] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test_kmeans)'''

    '''water_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    water_visualizer.fit(x_water_train, y_water_train)
    water_visualizer.show()

    kmeans_water = KMeans(n_clusters=6, random_state=0).fit(x_water_train, y_water_train)
    samples_water = x_water_train.sample(n=5, axis='columns', random_state=0)
    samples_water['Potability'] = y_water_train
    samples_water['class'] = kmeans_water.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''components_nasa = np.arange(1, 31)
    bic_nasa = np.zeros(components_nasa.size)

    for i, n in enumerate(components_nasa):
        gmm_nasa = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_nasa.fit(x_nasa_train, y_nasa_train)
        bic_nasa[i] = gmm_nasa.bic(x_nasa_train)

    plt.figure()
    plt.plot(components_nasa, bic_nasa)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('NASA: Asteroids Classification')
    plt.grid()
    plt.show()

    components_water = np.arange(1, 31)
    bic_water = np.zeros(components_water.size)

    for i, n in enumerate(components_water):
        gmm_water = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_water.fit(x_water_train, y_water_train)
        bic_water[i] = gmm_water.bic(x_water_train)

    plt.figure()
    plt.plot(components_water, bic_water)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('Water Potability')
    plt.grid()
    plt.show()'''

    '''gmm_nasa = GaussianMixture(n_components=5, reg_covar=1, random_state=0).fit(x_nasa_train, y_nasa_train)
    samples_nasa = x_nasa_train.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = gmm_nasa.predict(x_nasa_train)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''gmm_nasa = GaussianMixture(n_components=5, reg_covar=1, random_state=0).fit(x_nasa_train, y_nasa_train)
    x_nasa_train_gmm = gmm_nasa.predict_proba(x_nasa_train)
    x_nasa_test_gmm = gmm_nasa.predict_proba(x_nasa_test)
    
    y_nasa_mlppredict_train_gmm, y_nasa_mlppredict_test_gmm, nasa_train_time[2] = \
                             NNModel().mlpClassifier(x_nasa_train_gmm, x_nasa_test_gmm, \
                                                     y_nasa_train)

    nasa_train_accuracy[2] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train_gmm)
    nasa_test_accuracy[2] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test_gmm)

    classifiers = ('Original', 'K-Means', 'GM')
    y_pos = np.arange(len(classifiers))

    plt.figure()
    plt.barh(y_pos, nasa_test_accuracy)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_xlim(0.0, 1.0)
    plt.gca().set_yticklabels(classifiers)
    plt.gca().invert_yaxis()  # labels read top-to-bottom
    plt.title('Comparison of Accuracies')
    plt.xlabel('Accuracy')
    plt.subplots_adjust(left=0.15)
    plt.show()'''

    '''plt.figure()
    plt.barh(y_pos, nasa_train_time)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_yticklabels(classifiers)
    plt.gca().invert_yaxis()  # labels read top-to-bottom
    plt.title('Comparison of Training Time')
    plt.xlabel('Training Time (in seconds)')
    plt.subplots_adjust(left=0.15)
    plt.show()'''

    '''gmm_water = GaussianMixture(n_components=1, reg_covar=1, random_state=0).fit(x_water_train, y_water_train)
    samples_water = x_water_train.sample(n=5, axis='columns', random_state=0)
    samples_water['Potability'] = y_water_train
    samples_water['class'] = gmm_water.predict(x_water_train)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''pca_nasa = PCA(random_state=0)
    pca_nasa.fit(x_nasa_train, y_nasa_train)

    plt.figure()
    plt.plot(np.arange(1, pca_nasa.explained_variance_ratio_.size + 1), np.cumsum(pca_nasa.explained_variance_ratio_))
    plt.xticks(np.arange(1, pca_nasa.explained_variance_ratio_.size + 1))
    plt.xlabel('Component')
    plt.ylabel('Variance (cumulative)')
    plt.title('NASA: Asteroids Classification')
    plt.grid()
    plt.show()'''

    '''from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(x_water_train, y_water_train)

    pca_water = PCA(random_state=0)
    pca_water.fit(data_rescaled)

    plt.figure()
    plt.plot(np.arange(1, pca_water.explained_variance_ratio_.size + 1), np.cumsum(pca_water.explained_variance_ratio_))
    plt.xticks(np.arange(1, pca_water.explained_variance_ratio_.size + 1))
    plt.xlabel('Component')
    plt.ylabel('Variance (cumulative)')
    plt.title('Water Potability')
    plt.grid()
    plt.show()'''

    pca_nasa = PCA(n_components=2, random_state=0)
    x_nasa_transform = pca_nasa.fit_transform(x_nasa_train, y_nasa_train)

    '''nasa_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    nasa_visualizer.fit(x_nasa_transform, y_nasa_train)
    nasa_visualizer.show()'''

    '''samples_nasa = pd.DataFrame(x_nasa_transform)

    kmeans_nasa = KMeans(n_clusters=6, random_state=0).fit(x_nasa_transform, y_nasa_train)
    #samples_nasa = x_nasa_transform.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = kmeans_nasa.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''components_nasa = np.arange(1, 31)
    bic_nasa = np.zeros(components_nasa.size)

    for i, n in enumerate(components_nasa):
        gmm_nasa = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_nasa.fit(x_nasa_transform, y_nasa_train)
        bic_nasa[i] = gmm_nasa.bic(x_nasa_transform)

    plt.figure()
    plt.plot(components_nasa, bic_nasa)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('NASA: Asteroids Classification')
    plt.grid()
    plt.show()'''

    '''samples_nasa = pd.DataFrame(x_nasa_transform)
    
    gmm_nasa = GaussianMixture(n_components=6, reg_covar=1, random_state=0).fit(x_nasa_transform, y_nasa_train)
    #samples_nasa = x_nasa_train.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = gmm_nasa.predict(x_nasa_transform)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    pca_nasa = PCA(n_components=2, random_state=0)
    x_nasa_train_pca = pca_nasa.fit_transform(x_nasa_train, y_nasa_train)
    x_nasa_test_pca = pca_nasa.fit_transform(x_nasa_test, y_nasa_test)

    '''print(validation_curve(MLPClassifier(), x_nasa_train_pca, y_nasa_train.values.ravel(),\
                           'activation', ['identity', 'logistic', 'tanh', 'relu'],\
                           cv=10, scoring='accuracy'))
    print(validation_curve(MLPClassifier(), x_nasa_train_pca, y_nasa_train.values.ravel(),\
                           'hidden_layer_sizes', list(range(50, 150)),\
                           cv=10, scoring='accuracy'))'''

    y_nasa_mlppredict_train, y_nasa_mlppredict_test, nasa_train_time[0] = \
                             NNModel().mlpClassifier(x_nasa_train, x_nasa_test, y_nasa_train)
    
    y_nasa_mlppredict_train_pca, y_nasa_mlppredict_test_pca, nasa_train_time[1] = \
                             NNModel().mlpClassifier(x_nasa_train_pca, x_nasa_test_pca, \
                                                     y_nasa_train)

    nasa_train_accuracy[0] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train)
    nasa_test_accuracy[0] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test)

    nasa_train_accuracy[1] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train_pca)
    nasa_test_accuracy[1] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test_pca)

    pca_water = PCA(n_components=7, random_state=0)
    x_water_transform = pca_water.fit_transform(x_water_train, y_water_train)

    '''water_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    water_visualizer.fit(x_water_transform, y_water_train)
    water_visualizer.show()'''

    '''samples_water = pd.DataFrame(x_water_transform)

    kmeans_water = KMeans(n_clusters=6, random_state=0).fit(x_water_transform, y_water_train)
    samples_water['Hazardous'] = y_water_train
    samples_water['class'] = kmeans_water.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''components_water = np.arange(1, 31)
    bic_water = np.zeros(components_water.size)

    for i, n in enumerate(components_water):
        gmm_water = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_water.fit(x_water_transform, y_water_train)
        bic_water[i] = gmm_water.bic(x_water_transform)

    plt.figure()
    plt.plot(components_water, bic_water)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('Water Potability')
    plt.grid()
    plt.show()'''

    '''samples_water = pd.DataFrame(x_water_transform)
    
    gmm_water = GaussianMixture(n_components=1, reg_covar=1, random_state=0).fit(x_water_transform, y_water_train)
    samples_water['Hazardous'] = y_water_train
    samples_water['class'] = gmm_water.predict(x_water_transform)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    ica_nasa = FastICA(n_components=2, random_state=0)
    x_nasa_transform = ica_nasa.fit_transform(x_nasa_train, y_nasa_train)

    '''nasa_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    nasa_visualizer.fit(x_nasa_transform, y_nasa_train)
    nasa_visualizer.show()'''

    '''samples_nasa = pd.DataFrame(x_nasa_transform)

    kmeans_nasa = KMeans(n_clusters=7, random_state=0).fit(x_nasa_transform, y_nasa_train)
    #samples_nasa = x_nasa_transform.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = kmeans_nasa.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''components_nasa = np.arange(1, 31)
    bic_nasa = np.zeros(components_nasa.size)

    for i, n in enumerate(components_nasa):
        gmm_nasa = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_nasa.fit(x_nasa_transform, y_nasa_train)
        bic_nasa[i] = gmm_nasa.bic(x_nasa_transform)

    plt.figure()
    plt.plot(components_nasa, bic_nasa)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('NASA: Asteroids Classification')
    plt.grid()
    plt.show()'''

    '''samples_nasa = pd.DataFrame(x_nasa_transform)
    
    gmm_nasa = GaussianMixture(n_components=1, reg_covar=1, random_state=0).fit(x_nasa_transform, y_nasa_train)
    #samples_nasa = x_nasa_train.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = gmm_nasa.predict(x_nasa_transform)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    ica_nasa = FastICA(n_components=2, random_state=0)
    x_nasa_train_ica = ica_nasa.fit_transform(x_nasa_train, y_nasa_train)
    x_nasa_test_ica = ica_nasa.fit_transform(x_nasa_test, y_nasa_test)

    '''print(validation_curve(MLPClassifier(), x_nasa_train_ica, y_nasa_train.values.ravel(),\
                           'activation', ['identity', 'logistic', 'tanh', 'relu'],\
                           cv=10, scoring='accuracy'))
    print(validation_curve(MLPClassifier(), x_nasa_train_ica, y_nasa_train.values.ravel(),\
                           'hidden_layer_sizes', list(range(50, 150)),\
                           cv=10, scoring='accuracy'))'''

    y_nasa_mlppredict_train_ica, y_nasa_mlppredict_test_ica, nasa_train_time[2] = \
                             NNModel().mlpClassifier(x_nasa_train_ica, x_nasa_test_ica, \
                                                     y_nasa_train)

    nasa_train_accuracy[2] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train_ica)
    nasa_test_accuracy[2] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test_ica)

    ica_water = FastICA(n_components=7, random_state=0, tol=0.01)
    x_water_transform = ica_water.fit_transform(x_water_train, y_water_train)

    '''water_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    water_visualizer.fit(x_water_transform, y_water_train)
    water_visualizer.show()'''

    '''samples_water = pd.DataFrame(x_water_transform)

    kmeans_water = KMeans(n_clusters=11, random_state=0).fit(x_water_transform, y_water_train)
    samples_water['Hazardous'] = y_water_train
    samples_water['class'] = kmeans_water.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''
    
    '''components_water = np.arange(1, 31)
    bic_water = np.zeros(components_water.size)

    for i, n in enumerate(components_water):
        gmm_water = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_water.fit(x_water_transform, y_water_train)
        bic_water[i] = gmm_water.bic(x_water_transform)

    plt.figure()
    plt.plot(components_water, bic_water)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('Water Potability')
    plt.grid()
    plt.show()

    samples_water = pd.DataFrame(x_water_transform)
    
    gmm_water = GaussianMixture(n_components=1, reg_covar=1, random_state=0).fit(x_water_transform, y_water_train)
    samples_water['Hazardous'] = y_water_train
    samples_water['class'] = gmm_water.predict(x_water_transform)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''
    
    rp_nasa = GaussianRandomProjection(n_components=2, random_state=0)
    x_nasa_transform = rp_nasa.fit_transform(x_nasa_train, y_nasa_train)

    '''nasa_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    nasa_visualizer.fit(x_nasa_transform, y_nasa_train)
    nasa_visualizer.show()'''

    '''samples_nasa = pd.DataFrame(x_nasa_transform)

    kmeans_nasa = KMeans(n_clusters=7, random_state=0).fit(x_nasa_transform, y_nasa_train)
    #samples_nasa = x_nasa_transform.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = kmeans_nasa.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''components_nasa = np.arange(1, 31)
    bic_nasa = np.zeros(components_nasa.size)

    for i, n in enumerate(components_nasa):
        gmm_nasa = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_nasa.fit(x_nasa_transform, y_nasa_train)
        bic_nasa[i] = gmm_nasa.bic(x_nasa_transform)

    plt.figure()
    plt.plot(components_nasa, bic_nasa)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('NASA: Asteroids Classification')
    plt.grid()
    plt.show()'''

    '''samples_nasa = pd.DataFrame(x_nasa_transform)
    
    kmeans_nasa = GaussianMixture(n_components=6, reg_covar=1, random_state=0).fit(x_nasa_transform, y_nasa_train)
    #samples_nasa = x_nasa_train.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = kmeans_nasa.predict(x_nasa_transform)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    rp_nasa = GaussianRandomProjection(n_components=2, random_state=0)
    x_nasa_train_rp = rp_nasa.fit_transform(x_nasa_train, y_nasa_train)
    x_nasa_test_rp = rp_nasa.fit_transform(x_nasa_test, y_nasa_test)

    '''print(validation_curve(MLPClassifier(), x_nasa_train_rp, y_nasa_train.values.ravel(),\
                           'activation', ['identity', 'logistic', 'tanh', 'relu'],\
                           cv=10, scoring='accuracy'))
    print(validation_curve(MLPClassifier(), x_nasa_train_rp, y_nasa_train.values.ravel(),\
                           'hidden_layer_sizes', list(range(50, 150)),\
                           cv=10, scoring='accuracy'))'''

    y_nasa_mlppredict_train_rp, y_nasa_mlppredict_test_rp, nasa_train_time[3] = \
                             NNModel().mlpClassifier(x_nasa_train_rp, x_nasa_test_rp, \
                                                     y_nasa_train)

    nasa_train_accuracy[3] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train_rp)
    nasa_test_accuracy[3] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test_rp)

    rp_water = GaussianRandomProjection(n_components=7, random_state=0)
    x_water_transform = rp_water.fit_transform(x_water_train, y_water_train)

    '''water_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    water_visualizer.fit(x_water_transform, y_water_train)
    water_visualizer.show()

    samples_water = pd.DataFrame(x_water_transform)

    kmeans_water = KMeans(n_clusters=6, random_state=0).fit(x_water_transform, y_water_train)
    samples_water['Hazardous'] = y_water_train
    samples_water['class'] = kmeans_water.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''components_water = np.arange(1, 31)
    bic_water = np.zeros(components_water.size)

    for i, n in enumerate(components_water):
        gmm_water = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_water.fit(x_water_transform, y_water_train)
        bic_water[i] = gmm_water.bic(x_water_transform)

    plt.figure()
    plt.plot(components_water, bic_water)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('Water Potability')
    plt.grid()
    plt.show()'''

    '''samples_water = pd.DataFrame(x_water_transform)
    
    kmeans_water = GaussianMixture(n_components=1, reg_covar=1, random_state=0).fit(x_water_transform, y_water_train)
    samples_water['Hazardous'] = y_water_train
    samples_water['class'] = kmeans_water.predict(x_water_transform)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    #plt.ylim([-100,5000])
    plt.legend(loc='best')
    plt.show()'''

    sp_nasa = SelectPercentile(chi2, percentile=10)
    x_nasa_transform = sp_nasa.fit_transform(x_nasa_train, y_nasa_train)

    '''nasa_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    nasa_visualizer.fit(x_nasa_transform, y_nasa_train)
    nasa_visualizer.show()'''

    '''samples_nasa = pd.DataFrame(x_nasa_transform)

    kmeans_nasa = KMeans(n_clusters=6, random_state=0).fit(x_nasa_transform, y_nasa_train)
    #samples_nasa = x_nasa_transform.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = kmeans_nasa.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    '''components_nasa = np.arange(1, 31)
    bic_nasa = np.zeros(components_nasa.size)

    for i, n in enumerate(components_nasa):
        gmm_nasa = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_nasa.fit(x_nasa_transform, y_nasa_train)
        bic_nasa[i] = gmm_nasa.bic(x_nasa_transform)

    plt.figure()
    plt.plot(components_nasa, bic_nasa)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('NASA: Asteroids Classification')
    plt.grid()
    plt.show()

    samples_nasa = pd.DataFrame(x_nasa_transform)
    
    kmeans_nasa = GaussianMixture(n_components=4, reg_covar=1, random_state=0).fit(x_nasa_transform, y_nasa_train)
    #samples_nasa = x_nasa_train.sample(n=5, axis='columns', random_state=0)
    samples_nasa['Hazardous'] = y_nasa_train
    samples_nasa['class'] = kmeans_nasa.predict(x_nasa_transform)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_nasa, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('NASA: Asteroids Classification')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()'''

    sp_nasa = SelectPercentile(chi2, percentile=10)
    x_nasa_train_sp = sp_nasa.fit_transform(x_nasa_train, y_nasa_train)
    x_nasa_test_sp = sp_nasa.fit_transform(x_nasa_test, y_nasa_test)

    y_nasa_mlppredict_train_sp, y_nasa_mlppredict_test_sp, nasa_train_time[4] = \
                             NNModel().mlpClassifier(x_nasa_train_sp, x_nasa_test_sp, \
                                                     y_nasa_train)

    nasa_train_accuracy[4] = NNModel().mlpTrainAccuracy(y_nasa_train, y_nasa_mlppredict_train_sp)
    nasa_test_accuracy[4] = NNModel().mlpTestAccuracy(y_nasa_test, y_nasa_mlppredict_test_sp)

    classifiers = ('Original', 'PCA', 'ICA', 'RP', 'SP')
    y_pos = np.arange(len(classifiers))

    plt.figure()
    plt.barh(y_pos, nasa_train_accuracy)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_xlim(0.0, 1.0)
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

    sp_water = SelectPercentile(chi2, percentile=10)
    x_water_transform = sp_water.fit_transform(x_water_train, y_water_train)

    '''water_visualizer = KElbowVisualizer(KMeans(random_state=0), k=(2,31))
    water_visualizer.fit(x_water_transform, y_water_train)
    water_visualizer.show()'''

    '''samples_water = pd.DataFrame(x_water_transform)

    kmeans_water = KMeans(n_clusters=6, random_state=0).fit(x_water_transform, y_water_train)
    samples_water['Hazardous'] = y_water_train
    samples_water['class'] = kmeans_water.labels_

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    #plt.ylim([-1,5])
    plt.legend(loc='best')
    plt.show()

    components_water = np.arange(1, 31)
    bic_water = np.zeros(components_water.size)

    for i, n in enumerate(components_water):
        gmm_water = GaussianMixture(n_components=n, reg_covar=1, random_state=0)
        gmm_water.fit(x_water_transform, y_water_train)
        bic_water[i] = gmm_water.bic(x_water_transform)

    plt.figure()
    plt.plot(components_water, bic_water)
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('Water Potability')
    plt.grid()
    plt.show()

    samples_water = pd.DataFrame(x_water_transform)
    
    kmeans_water = GaussianMixture(n_components=2, reg_covar=1, random_state=0).fit(x_water_transform, y_water_train)
    samples_water['Hazardous'] = y_water_train
    samples_water['class'] = kmeans_water.predict(x_water_transform)

    plt.figure()
    pd.plotting.parallel_coordinates(samples_water, 'class', colormap='Set1')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Water Potability')
    plt.tight_layout()
    #plt.ylim([-100,5000])
    plt.legend(loc='best')
    plt.show()'''

if __name__ == "__main__":
    main()
