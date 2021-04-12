from abc import ABC, abstractmethod
from collections import deque

class ITree(ABC):
    @abstractmethod
    def add_layer(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class GBDTTree(ITree):
    def __init__(self):
        self.tree = [] 

    def add_layer(self, feature,  iscaterical, lnode, rnode, split_value):
        self.tree.append([feature, iscaterical, lnode, rnode, split_value])

    def __str__(self):
        print_str = ''
        for i, (ft, _, l, r, v) in enumerate(self.tree, start = 1):
            print_str = "".join((print_str, f'{i}st tree, featuret: {ft}, left_node: {l}, right_node:{r}, split_value:{v} \n'))
        return print_str

    def predict(self, data):
        """
        can only predict one row data once a time
        """
        output = 0
        for ft, iscaterical, lnode, rnode, split_value in self.tree:
            if iscaterical:
                output = output + lnode if data[ft] == split_value else output + rnode
            else:
                output = output + lnode if data[ft] <= split_value else output + rnode
        return output



class ID3Tree(ITree):
    def __init__(self):
        self.root = None

    def add_layer(self, node):
        self.root = node

    def __str__(self):
        pass

    def predict(self, data):
        node = self.root
        while node.split_feature:
            ft = node.split_feature
            value = data[ft][0]
            print(ft, value)
            node = node.next_node_dict[str(value)]
            output = node.label
        return output

class TreeNode():
    def __init__(self, value=None):
        self.split_feature = None
        self.label = None
        self.next_node_dict = {}
        self.is_leaf = False
        self.value = value
        self.layer_num = 1

    def set(self, feature, values):
        self.split_feature = feature 
        for v in values:
            self.next_node_dict[str(v)] = None

    def add_next_node(self, value, node):
        self.next_node_dict[str(value)] = node
        node.layer_num = self.layer_num + 1

    def set_output_label(self, label):
        self.label = label
        self.is_leaf = True

