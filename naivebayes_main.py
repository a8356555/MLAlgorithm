import numpy as np
from dataset import Dataset
from NaiveBayesModel import NaiveBayesClassifier

def main():
    dtypes = [('handsome', 'S10'),
             ('character', 'S10'),
             ('height', 'S10'),
             ('attitude', 'S10'),
             ('married', np.uint8)]

    data = np.array([('handsome', 'bad', 'short', 'negative', 0),
                    ('not_handsome', 'good', 'short', 'attentive', 0),
                    ('handsome', 'good', 'short', 'attentive', 1),
                    ('not_handsome', 'good', 'tall', 'attentive', 1),
                    ('handsome', 'bad', 'short', 'attentive', 0),
                    ('not_handsome', 'bad', 'short', 'negative', 0),
                    ('handsome', 'good', 'tall', 'negative', 1),
                    ('not_handsome', 'good', 'tall', 'attentive', 1),
                    ('handsome', 'good', 'tall', 'attentive', 1),
                    ('not_handsome', 'bad', 'tall', 'attentive', 1),
                    ('handsome', 'good', 'short', 'negative', 0),
                    ('handsome', 'good', 'short', 'negative', 0)], dtype = dtypes)


    dtypes2 = [('height', np.float_),
               ('weight', np.float_),
               ('foot_feet', np.float_),
               ('gender', 'S10')]
    data2 = np.array([(6, 180, 12, 'M'),
                      (5.92, 190, 11, 'M'),
                      (5.58, 170, 12, 'M'),
                      (5.92, 165, 10, 'M'),
                      (5, 100, 6, 'F'),
                      (5.5, 150, 8, 'F'),
                      (5.42, 130, 7, 'F'),
                      (5.75, 150, 9, 'F')], dtype = dtypes2)

    model = NaiveBayesClassifier()
    data = Dataset(data, 'married')

    x = ('not_handsome', 'bad', 'short', 'negative')
    x = np.array(x, dtype = dtypes[:-1])
    
    model.fit(data)
    model.predict(x)
    

    model2 = NaiveBayesClassifier()
    data2 = Dataset(data2, 'gender')

    x2 = (6, 130, 8)
    x2 = np.array(x2, dtype = dtypes2[:-1])
    
    model2.fit(data2)
    model2.predict(x2)
    
if __name__ == "__main__":
    main()
