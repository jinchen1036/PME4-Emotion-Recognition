import numpy as np
import pandas as pd

# loading models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def generate_record(filetype, model_type, accuracy, n_coef):
    print(f"{filetype} - {model_type} - accuracy = {'%.5f' % accuracy}")
    return {
        "filetype": filetype,
        "model_type": model_type,
        "accuary": accuracy,
        "n_coef": n_coef
    }


def train_traditional_model(filetype, train_data, test_data, train_labels, test_labels, n_coef=None):
    if n_coef is not None:
        train_data = train_data[:, :, :n_coef]
        test_data = test_data[:, :, :n_coef]
    else:
        n_coef = train_data.shape[-1]
    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    test_data = np.reshape(test_data, (test_data.shape[0], -1))
    train_labels = np.reshape(train_labels, (-1))
    test_labels = np.reshape(test_labels, (-1))
    print("Train shape:", train_data.shape)
    print("Test shape:", test_data.shape)

    traditional_model_results = []
    # knn
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance', p=2, metric='minkowski')
    knn.fit(train_data, train_labels)
    knn_acc = knn.score(test_data, test_labels)
    traditional_model_results.append(generate_record(filetype=filetype, model_type="KNN-10", accuracy=knn_acc, n_coef=n_coef))

    # SVM
    svm = SVC(kernel='rbf')
    svm.fit(train_data, train_labels)
    svm_acc = svm.score(test_data, test_labels)
    traditional_model_results.append(generate_record(filetype=filetype, model_type="SVM_RBF", accuracy=svm_acc, n_coef=n_coef))

    # Random Forest
    random_forest = RandomForestClassifier(max_depth=7, n_estimators=100, max_features='sqrt')
    random_forest.fit(train_data, train_labels)
    rf_acc = random_forest.score(test_data, test_labels)
    traditional_model_results.append(
        generate_record(filetype=filetype, model_type="RF_100N_7MD", accuracy=rf_acc, n_coef=n_coef))

    # MLP
    mlp = MLPClassifier(random_state=42, hidden_layer_sizes=512, max_iter=100,
                          learning_rate='adaptive', alpha=1)
    mlp.fit(train_data, train_labels)
    mlp_acc = mlp.score(test_data, test_labels)
    traditional_model_results.append(
        generate_record(filetype=filetype, model_type="MLP_512NH_100MI", accuracy=mlp_acc, n_coef=n_coef))

    traditional_model_results_df = pd.DataFrame(traditional_model_results)
    return traditional_model_results_df
