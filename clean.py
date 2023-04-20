import pandas as pd
import numpy as np


def is_outside_threshold(value, mean, threshold):
	 return value > mean * (1 + threshold) or value < mean * (1 - threshold)


def clean_features(feature_names=None):
	data_features_path = "data/raw/machine_parameters.xlsx"
	features_df = pd.read_excel(data_features_path)

	if feature_names == None:
		feature_names = ["Cykler", "Materialepude", "Sprøjtetid", "Doséringstid", "Cyklustid",
			"Omskiftning til eftertryk", "Sprøjtetryk maks.", "Sprøjtetryk Integral"]

	return features_df[feature_names].dropna()


def clean_labels():
	data_labels_path = "data/raw/our_dataset_labels.xlsx"
	labels_df = pd.read_excel(data_labels_path)

	label_names = ["weight1 (no dot)", "weight2 (dot)", "dimension (no dot)", "dimension (dot)"]
	means = {label_name: labels_df[label_name].mean() for label_name in label_names}

	labels = {name: [] for name in label_names}
	threshold = 0.01
	for idx, row in labels_df.iterrows():
		for name in label_names:
			mean = means[name]
			value = row[name]
			cycle = row["cycle"]

			label = not is_outside_threshold(value, mean, threshold)
			
			labels[name].append([int(cycle), label])

	return list(labels.values())


def get_cleaned_features_and_labels(feature_names=None):
	features = clean_features(feature_names)
	labels = clean_labels()

	# align them
	features = features.sort_values(by=["Cykler"], ascending=True).to_numpy()
	labels = np.array(labels).reshape(212, 4, 2)

	# remove the columns with cycles, it is only meant for debugging
	features = np.delete(features, 0, 1)
	labels = np.delete(labels, 0, 2)

	return features, labels


def main():
	features, labels = get_cleaned_features_and_labels()

	np.save("data/processed/features", features)
	np.save("data/processed/labels", labels)


if __name__ == "__main__":
	main()