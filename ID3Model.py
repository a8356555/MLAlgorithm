import numpy as np 
from Tree import TreeNode
import math
from collections import Counter

class ID3TreeMaker:
    def __init__(self):
        pass

    def get_info_entropy(self, labels):
        """
        caculate ie info_entropy
        Px = Dk/D
        ie = -sum(Px1*log2(Px1)+...)

        """
        D = len(labels)
        info_entropy= 0
        for label in np.unique(labels):
            Px = sum(labels == label)/D
            info_entropy -= Px*math.log2(Px)

        return info_entropy

    def get_cond_entropy(self, data, feature, labels):
        """
        calculate ce condition_entropy

        """
        D = len(data)
        cond_entropy = 0
        for ft in np.unique(data[feature]):
            di = labels[data[feature] == ft]
            entropy = self.get_info_entropy(di)
            cond_entropy += len(di)/D*entropy
        return cond_entropy

    def find_feature(self, data, features, labels):
        """
        find min info gain = find min conditional entropy
        """
        min_cond_entropy = float('inf')
        best_feature = None
        # info_entropy = self.H_d(data.labels)
        # print(f'H(D):{info_entropy}')
        for ft in features:
            temp_cond_entropy = self.get_cond_entropy(data, ft, labels)
            if temp_cond_entropy < min_cond_entropy:
                min_cond_entropy = temp_cond_entropy
                best_feature = ft
        return best_feature

    def make_tree(self, data, features, labels, model):
        node = TreeNode()
        model.add_layer(node)
        self.add_branch(data, features, labels, node)
        return node

    def add_branch(self, data, features, labels, node):
        feature = self.find_feature(data, features, labels)
        feature_values = np.unique(data[feature])
        node.set(feature, feature_values)
        for value in feature_values: 
            subset_cond = data[feature] == value
            subset_data = data[subset_cond]
            subset_labels = labels[subset_cond]
            subset_features = [ft for ft in features if ft != feature] 
            new_node = TreeNode(value)
            node.add_next_node(value, new_node)
            counter = Counter(subset_labels)
            label = counter.most_common(1)[0][0]
            new_node.set_output_label(label)
            if len(np.unique(subset_labels)) > 1 and len(subset_features) > 0:
                self.add_branch(subset_data, subset_features, subset_labels, new_node)




