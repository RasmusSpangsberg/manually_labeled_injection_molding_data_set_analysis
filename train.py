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
from sklearn.dummy import DummyClassifier

# utils
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import normalize, MinMaxScaler
import json



def create_train_test_split_from_experiment_classes():
    # we did 7 different experiments, make sure no data from one leaks into the other
    experiment_ranges = [range(1, 52), range(52, 77), range(77, 103), range(103, 127),
        range(127, 146), range(146, 170), range(170, 213)]

    # to get different train/test splits
    experiment_ranges = shuffle(experiment_ranges)

    return experiment_ranges[:5], experiment_ranges[5:]


features = np.load("data/processed/features.npy", allow_pickle=True)
labels = np.load("data/processed/labels.npy", allow_pickle=True)




'''
train_ranges, test_ranges = create_train_test_split_from_experiment_classes()

features_train, labels_train, features_test, labels_test = [], [], [], []

for train_range in train_ranges:
    start_idx = min(train_range)
    end_idx = max(train_range)
    
    features_train.append(features[start_idx:end_idx])
    labels_train.append(labels[start_idx:end_idx])

for test_range in test_ranges:
    start_idx = min(test_range)
    end_idx = max(test_range)
    
    features_test.append(features[start_idx:end_idx])
    labels_test.append(labels[start_idx:end_idx])

features_train = np.concatenate(features_train)
labels_train = np.concatenate(labels_train)
features_test = np.concatenate(features_test)
labels_test = np.concatenate(labels_test)
'''

#features, labels = shuffle(features, labels)

'''
print(features[0])
#features = normalize(features)
#features = MinMaxScaler().fit_transform(features)
print(features[0])
'''

label_names = ["weight1 (no dot)", "weight2 (dot)", "dimension (no dot)", "dimension (dot)"]

num_experiments = 5


#clf_models = [DummyClassifier(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]#, SVC(), GradientBoostingClassifier()]
clf_models = [DummyClassifier(), RandomForestClassifier()]

metrics = ["bad_samples_caught", "good_samples_discarded", "acc", "f1"]
results = {label_name: {str(model): {metric: [] for metric in metrics} for model in clf_models} for label_name in label_names}

for _ in range(num_experiments):
    for i in range(4):
        #print(" -----------------", label_names[i], "------------------------ ")
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels[:, i].ravel(), test_size=0.30)

        '''
        for model in clf_models:
            model.fit(features_train, labels_train)
            
            preds = model.predict(features_test)
            acc = accuracy_score(labels_test, preds)
            f1 = f1_score(labels_test, preds)

            for i in range(len(preds)):
                if pred[i] != 

            print(f"{model}, {acc:.2f}")
            print(f"{model}, {f1:.2f}")
        '''

        # good and bad samples discarded
        for model in clf_models:
            model.fit(features_train, labels_train)
            
            preds = model.predict(features_test)
            tn, fp, fn, tp = confusion_matrix(labels_test, preds).ravel()

            '''
            print()
            print(model)
            print(f"Bad  samples caught:    {tn}/{tn+fp} = {tn/(tn+fp)*100:.0f}%")
            print(f"Good samples discarded: {fn}/{fn+tp} = {fn/(fn+tp)*100:.0f}%")
            '''

            acc = accuracy_score(labels_test, preds)
            f1 = f1_score(labels_test, preds)

            label_name = label_names[i]
            model_name = str(model)
            results[label_name][model_name]["bad_samples_caught"].append(tn/(tn+fp))
            results[label_name][model_name]["good_samples_discarded"].append(fn/(fn+tp))
            results[label_name][model_name]["acc"].append(acc)
            results[label_name][model_name]["f1"].append(f1)

# turn list of metric values into the mean of the list
for label_name, models in results.items():
    for model, metrics in models.items():
        for metric, value in metrics.items():
            results[label_name][model][metric] = round(np.mean(value), 2)

print(json.dumps(results, indent=2))