import numpy as np

# models
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier

# utils
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score
from sklearn.preprocessing import normalize, MinMaxScaler

from clean import get_cleaned_features_and_labels

def load_data1():
    feature_names = ["Cykler", "Materialepude", "Doséringstid", "Cyklustid", "Sprøjtetryk maks."]

    features, labels = get_cleaned_features_and_labels(feature_names)

    # only take the first column, which is weight1
    labels = labels[:, 0].ravel()

    features, labels = shuffle(features, labels)

    return features, labels

def load_data2():
    features = np.load(r"C:\repos\quality_prediction\data\processed\features.npy", allow_pickle=True)
    labels = np.load(r"C:\repos\quality_prediction\data\processed\labels.npy", allow_pickle=True)

    features, labels = shuffle(features, labels)
    labels = labels.ravel()

    return features,  labels


features1, labels1 = load_data1()
features2, labels2 = load_data2()

'''
print(features[0])
#features = normalize(features)
#features = MinMaxScaler().fit_transform(features)
print(features[0])
'''

#clf_models = [DummyClassifier(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]
clf_models = [DummyClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]

# cross validation
print("Cross validation without transferlearning")
for model in clf_models:
    accuracy = cross_val_score(model, features1, labels1, cv=5)

    print(f"{model}, {accuracy.mean():.2f}")

# transferlearning
features_train, features_test, labels_train, labels_test = train_test_split(features1, labels1,
    test_size=0.20)

features_train_combined = np.concatenate((features_train, features2))
labels_train_combined = np.concatenate((labels_train, labels2))

# value the data set that we try transfer learning to, 4 times more than the other
sample_weight = [4 if i < len(labels_train) else 1 for i in range(len(labels_train_combined))]

features_train_combined, labels_train_combined, sample_weight = shuffle(features_train_combined, labels_train_combined, sample_weight)

print("\n - With transferlearning")
for model in clf_models:
    model.fit(features_train_combined, labels_train_combined, sample_weight=sample_weight)
    
    preds = model.predict(features_test)
    acc = accuracy_score(labels_test, preds)

    print(f"{model}, {acc:.2f}")

print("\n - Without transferlearning")
for model in clf_models:
    model.fit(features_train, labels_train)
    
    preds = model.predict(features_test)
    acc = accuracy_score(labels_test, preds)

    print(f"{model}, {acc:.2f}")


# ------ for examining how much data is needed for a certain amount of accuracy
print("\n - Accuracy pr. num data samples")
for data_amount in [.05, .1, .2, .4, .5, .7, .9, 1]:
    end_idx = round(len(features2) * data_amount)

    features2_reduced = features2[:end_idx]
    labels2_reduced = labels2[:end_idx]

    model = RandomForestClassifier()
    accuracy = cross_val_score(model, features2_reduced, labels2_reduced, cv=5)

    print(f"num samples: {end_idx}, accuracy: {accuracy.mean():.2f}")