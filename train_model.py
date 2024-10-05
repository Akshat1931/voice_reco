import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
import preprocess_audio

def load_data(directory):
    """
    Loads the audio dataset, processes it, and returns features and labels.
    """
    labels = []
    features = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            label = filename.split('_')[0]  # Assuming the label is part of the filename
            mfcc = preprocess_audio.preprocess_audio(os.path.join(directory, filename))
            if mfcc is not None:  # Only append if the preprocessing was successful
                features.append(mfcc)
                labels.append(label)
    return np.array(features), np.array(labels)

# Load your dataset
features, labels = load_data('dataset')
print("Labels:", np.unique(labels))


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM model
model = svm.SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the model and scaler
joblib.dump(model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
