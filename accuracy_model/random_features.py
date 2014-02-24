import collections
import pickle

import numpy as np


class RandomFeatures:

    def __init__(self, num_features=50):
        self.dynamic_mode = True
        self.random_components = collections.defaultdict(
            self._generate_component)
        self.num_features = num_features
        self.reset_features()

    def _generate_component(self):
        rv = np.random.randn(self.num_features, 1)
        rv /= np.sqrt(np.dot(rv.T, rv))  # normalize to unit length
        return rv

    def load_components(self, filename):
        with open(filename, 'r') as f:
            self.set_components(pickle.load(f))

    def set_components(self, component_dict):
        # TODO(jace) assert all component vectors have same length
        self.num_features = component_dict[component_dict.keys()[0]].size
        self.random_components = component_dict
        self.dynamic_mode = False
        self.reset_features()

    def write_components(self, filename):
        with open(filename, 'w') as f:
            # conversion of defaultdict to dict is advisable for pickling
            pickle.dump(dict(self.random_components), f)

    def contains_component(self, component_key):
        return component_key in self.random_components

    def reset_features(self):
        self.feature_vector = np.zeros((self.num_features, 1))

    def get_features(self):
        return self.feature_vector[:, 0]

    def increment_component(self, component_key, scale=1.0):
        if not self.dynamic_mode and (
                component_key not in self.random_components):
            return False

        self.feature_vector += scale * self.random_components[component_key]
        return True


if __name__ == '__main__':
    # Example usage
    rf = RandomFeatures(5)
    rf.increment_component("monkey")
    print rf.feature_vector
