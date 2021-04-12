import numpy as np
from Tree import GBDTTree
from GBDTModel import GBDTRegressorMaker
from dataset import Dataset


def main():
    print('start!')
    dtype = [('x', np.int8), ('y', np.float16)]
    data = np.array([(1, 5.56), 
                     (2, 5.7),
                     (3, 5.91),
                     (4, 6.4),
                     (5, 6.8),
                     (6, 7.05),
                     (7, 8.9), 
                     (8, 8.7),
                     (9,  9), 
                     (10, 9.05)], dtype=dtype)
    max_depth = 5
    data = Dataset(data, 'y')
    gbdt = GBDTRegressorMaker(5)
    model = GBDTTree()
    gbdt.make_tree(data, model)
    print(model)
    
    test = np.array([8], dtype=[('x', np.int8)])
    print(model.predict(test))

if __name__ == "__main__":
    main()
