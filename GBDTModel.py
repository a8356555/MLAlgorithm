import numpy as np

class GBDTRegressorMaker:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def square_error(self, y):
        mean = np.mean(y)
        se = 0
        for elem in y:
            se += (elem - mean)**2
        return se, mean

    def iscategorical(self, col):
        if len(col) > len(np.unique(col))/2:
            return True
        if isinstance(col[0], str):
            return True
        return False

    def find_feature(self, data):
        final_min_se = float('inf')
        for ft in data.features:
            min_se, final_lval, final_rval, final_split_value = self.find_split(data, ft, final_min_se) 
            if min_se < final_min_se:
                final_min_se = min_se
                final_feature = ft
        return final_feature, final_lval, final_rval, final_split_value

    def find_split(self, data, ft, final_min_se):
        iscategorical = data.iscategory[data.features == ft] 
        for split_value in np.unique(data[ft]): 
            left = data.labels[data[ft] == split_value] if iscategorical else data.labels[data[ft] <= split_value]
            right = data.labels[data[ft] != split_value] if iscategorical else data.labels[data[ft] > split_value]
            lse, lmean = self.square_error(left)
            rse, rmean = self.square_error(right)
            if lse + rse <= final_min_se:
                final_min_se = lse + rse
                final_lval, final_rval = lmean, rmean
                final_split_value = split_value 

        return final_min_se, final_lval, final_rval, final_split_value

    def residual(self, data, feature, lval, rval, value):
        iscategorical = data.iscategory[data.features == feature] 
        minor = (data[feature] == value) * lval + (data[feature] != value) * rval if iscategorical else (data[feature] <= value) * lval + (data[feature] > value) * rval
        data.labels = data.labels - minor
        return iscategorical, data
    
    def make_tree(self, data, model):
        for _ in range(self.max_depth):
            feature, lval, rval, value= self.find_feature(data)
            iscategorical, data = self.residual(data, feature, lval, rval, value)

            model.add_layer(feature, iscategorical, lval, rval, value)

