import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


class Svm:
    # c is the switcher of hardmargin and softmargin, default value 0 means hardmargin
    # otherwise its value is the indicator of slack

    def __init__(self, x, y, gamma=1, c=0):
        self.c = c
        self.gamma = gamma
        self.x = x#np.array(x)
        self.y = np.array(y)
        self.b = 0
        self.alpha = []
        #num of samples
        self.m = np.shape(self.x)[0]
        self.x_param_max = np.ones(np.shape(self.x)[1])
        self.x_param_min = np.ones(np.shape(self.x)[1])

        self.prediction_results = None
        self.test_results = None

        temp_x = np.ones(np.shape(self.x.T))
        # get max and min of each parameter so we can scale them to
        for i, rows in enumerate(self.x.T):
            min_val= min(rows)
            max_val = max(rows)

            self.x_param_min[i] = min_val
            self.x_param_max[i] = max_val

            difference = max_val-min_val
            if difference == 0:
                difference += .000001

            temp_x[i] = np.divide(np.subtract(rows, self.x_param_min[i]), difference)

        self.x = temp_x.T*2 -1

        self.fit()

    def scale(self, rows):
        # transform data to fit between [0,1] using max/mins from test data, then linearly transform to a range [-1,1]
        scaler = StandardScaler()
        return scaler.fit_transform(rows)

    def fit(self):
        print "Calculating coefficients"
        alpha_0 = np.ones(self.m)

        cons = ({'type': 'eq', 'fun': lambda a: np.dot(a, self.y)},
                # 2nd constraint: 0 < a < c
                {'type': 'ineq', 'fun': lambda a: self.c-a},
                {'type': 'ineq', 'fun': lambda a: a})

        def objective(a):
            temp = 0
            # calculate the sum of the products of the outer class and weight, inner class and weight
            # and the kernel with the outer and inner parameters
            for outer_a, outer_x, outer_y in zip(a, self.x, self.y):
                for inner_a, inner_x, inner_y in zip(a, self.x, self.y):
                    temp += outer_a*outer_y*self.gaussianKernel(outer_x, inner_x)*inner_a*inner_y
            # want to maximize, not minimize, so change the sign for the function
            return -(np.sum(a) - .5*temp)

        # get the optimal coefficients
        self.alpha = minimize(objective, alpha_0, constraints=cons, method='SLSQP').x

        # calculate the intercept
        self.b = self.multiply_kernel(self.x[0]) - self.y[0]

        return self

    def predict(self, x, y=None):
        x_scaled = self.scale(x)
        y = np.zeros(np.shape(x_scaled)[0])
        for index, sample in enumerate(x_scaled):
            if self.multiply_kernel(sample) + self.b > 0:
                y[index] = 1
            else:
                y[index] = -1
        return y

    def multiply_kernel(self, x):
        temp = 0
        for outer_a, outer_x, outer_y in zip(self.alpha, self.x, self.y):
            temp += outer_a*outer_y * self.gaussianKernel(outer_x, x)
        return temp

    def gaussianKernel(self, X, Y=0):
        return np.exp(-self.gamma * np.linalg.norm(np.subtract(X,Y) ** 2))

    def max_function(self, a, x, y):
        temp = 0
        for outer_a, outer_x, outer_y in zip(a,x,y):
            for inner_a, inner_x, inner_y in zip(a,x,y):
                temp += outer_a*inner_a*outer_y*inner_y*self.gaussianKernel(outer_x, inner_x)
        return np.sum(a) - temp/2

    def score(self, x, y):
        self.prediction_results = self.predict(x)
        self.test_results = y
        # subtract our results with the actual. If they match, it will become 0
        # 1 - the count of non zero elements and divide by the total number of test samples for the accuracy
        y = np.hstack(y)

        return 1-(np.count_nonzero(np.subtract(self.prediction_results,y))*1.0 / np.size(y))


class SvmFreeParam:
    def __init__(self, x, y, gamma_val=None, c_val=None):
        self.x = x
        self.y = y

        num_samples = np.shape(self.x)[0]
        test_amt = int(num_samples*.9)

        self.sample_x = self.x[0:test_amt]
        self.sample_y = self.y[0:test_amt]

        self.test_x = self.x[test_amt:]
        self.test_y = self.y[test_amt:]

        if gamma_val is None:
            self.gamma_list = gamma_range = [.01, 1]
        else:
            self.gamma_list = gamma_val

        if c_val is None:
            self.c_list = [.01, 1]
        else:
            self.c_list = c_val

        self.best_svm = None


    def svm_test(self, test_x, test_y):
        self.best_svm.score(test_x, test_y)

    def optimize_params(self):
        print "---Optimizing---"
        max_accuracy = 0
        for gamma in self.gamma_list:
            for c in self.c_list:
                test = Svm(self.sample_x, self.sample_y, gamma=gamma, c=c)
                print "Testing algorithm"
                cur_score = test.score(self.test_x, self.test_y)
                if cur_score > max_accuracy:
                    max_accuracy = cur_score
                    self.best_svm = test
                    print "Best accuracy is now: ", max_accuracy
        return max_accuracy
