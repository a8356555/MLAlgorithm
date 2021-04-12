import math
import numpy as np
from collections import defaultdict

def zero():
    return 0

class NaiveBayesClassifier:
    def __init__(self):
        """
        key : label_labelvalue_featurename_featurevalue
        """

        self.count = defaultdict(zero)
        self.labels = {}
        self.mean = {}
        self.variance = {}
        self.features = []
        self.iscategory = []

    def fit(self, data):
        """
        2.3 times faster than (for loop one row, just loop once)
        """
        self.labels = np.unique(data.labels)
        self.features = data.features
        self.iscategory = data.iscategory

        for label in self.labels:
            self.count[''.join(('label', '_', str(label)))] = sum(data.labels == label)

        for ft, iscategory in zip(data.features, data.iscategory):
            if iscategory:
                for value in np.unique(data[ft]):
                    for label in self.labels:
                        label_ft_comb = "".join(('label', '_', str(label), '_', str(ft), '_', str(value)))
                        self.count[label_ft_comb] = len(data[(data.labels == label) & (data[ft] == value)])
                        
            else:
                for label in self.labels:
                    label_ft_comb = "".join(('label', '_', str(label), '_', str(ft)))
                    data_given_label_ft_comb = data[ft][data.labels == label] 
                    self.mean[label_ft_comb] = np.mean(data_given_label_ft_comb)
                    self.variance[label_ft_comb] = np.var(data_given_label_ft_comb)
        print(self.count, self.mean, self.variance)

    def predict(self, x):
        min_p = 0
        pred = None
        for label in self.labels:
            p = self.count[''.join(('label', '_', str(label)))]
            for ft, iscategory in zip(self.features, self.iscategory):
                if iscategory:
                    label_ft_comb = "".join(('label', '_', str(label), '_', str(ft), '_', str(x[ft])))
                    p_ft_given_label = self.count[label_ft_comb]/self.count[''.join(('label', '_', str(label)))]
                    p *= p_ft_given_label
                else:
                    label_ft_comb = "".join(('label', '_', str(label), '_', str(ft)))
                    a = self.get_gaussian_prob(label_ft_comb, x[ft])
                    p = p*a
            if p > min_p:
                min_p = p
                pred = label
        return pred

    def get_gaussian_prob(self, label_ft_comb, x_ft):
        mean = self.mean[label_ft_comb]
        var = self.variance[label_ft_comb]
        prob = 1/math.sqrt(2*math.pi*var) * math.exp(-(x_ft-mean)**2/(2*var))
        return prob

