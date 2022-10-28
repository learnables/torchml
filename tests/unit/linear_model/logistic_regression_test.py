import unittest
import numpy as np
import torch
import torchml as ml
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


class TestLogisticRegression(unittest.TestCase):
    def test_fit(self):
        # Generate data (binary classification)
        separable = False
        while not separable:
            samples = make_classification(
                n_samples=1000,
                n_features=2,
                n_redundant=0,
                n_informative=1,
                n_clusters_per_class=1,
                flip_y=-1,
            )
            red = samples[0][samples[1] == 0]
            blue = samples[0][samples[1] == 1]
            separable = any(
                [
                    red[:, k].max() < blue[:, k].min()
                    or red[:, k].min() > blue[:, k].max()
                    for k in range(2)
                ]
            )

        red_labels = np.zeros(len(red))
        blue_labels = np.ones(len(blue))

        labels = np.append(red_labels, blue_labels)
        inputs = np.concatenate((red, blue), axis=0)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            inputs, labels, test_size=0.33, random_state=42
        )

        # Compare sklearn and torchml results
        ref = linear_model.LogisticRegression()
        ref.fit(X_train, y_train)
        ref_preds = ref.predict(X_test)

        model = ml.linear_model.LogisticRegression()
        model.fit(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.float32)),
        )
        model_preds = model.predict(torch.from_numpy(X_test.astype(np.float32)))

        self.assertTrue(np.allclose(ref_preds, model_preds))


if __name__ == "__main__":
    unittest.main()
