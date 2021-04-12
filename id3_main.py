import numpy as np
import math
from ID3Model import ID3TreeMaker
from dataset import Dataset
from Tree import ID3Tree

def main():
    dtypes = [('outlook', 'S10'), ('temperature', 'S10'), ('humidity', 'S10'), ('windy', 'S10'), ('play', 'S5')]
    data = np.array([('sunny', 'hot', 'high', 'False', 'no'),
                     ('sunny', 'hot', 'high', 'True', 'no'),
                     ('overcast', 'hot', 'high', 'False', 'yes'),
                     ('rainy', 'mild', 'high', 'False', 'yes'),
                     ('rainy', 'cool', 'normal', 'False', 'yes'),
                     ('rainy', 'cool', 'normal', 'True', 'no'),
                     ('overcast', 'cool', 'normal', 'True', 'yes'),
                     ('sunny', 'mild', 'high', 'False', 'no'),
                     ('sunny', 'cool', 'normal', 'False', 'yes'),
                     ('rainy', 'mild', 'normal', 'False', 'yes'),
                     ('sunny', 'mild', 'normal', 'True', 'yes'),
                     ('overcast', 'mild', 'high', 'True', 'yes'),
                     ('overcast', 'hot', 'normal', 'False', 'yes'),
                     ('rainy', 'mild', 'high', 'True', 'no')] , dtype=dtypes)

    data = Dataset(data, 'play')
    id3 = ID3TreeMaker()
    model = ID3Tree()
    id3.make_tree(data, data.features, data.labels, model)
    # print(model)
    x = np.array([('sunny', '123', 'high', 'False')], dtype=dtypes[:-1])
    pred = model.predict(x)
    print(pred)



if __name__ == "__main__":
    main()
