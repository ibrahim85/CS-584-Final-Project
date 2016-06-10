
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


class AlgoEval:
    def __init__(self, predicted=None, actual=None):
        self.predicted_results = predicted
        self.actual_results = actual


        self.confusion_matrices = None

        # stores data for each class
        self.true_positive = None
        self.false_positive = None
        self.true_negative = None
        self.false_negative = None

    def print_evaluations(self):
        self.evaluate_problems()
        print "\n----------------------"
        print "Confusion matrix:"
        print self.confusion_matrices

        print "\n----------------------"
        print "Accuracy:"
        print self.accuracy()

        print "\n----------------------"
        print "Classes that each tuple corresponds to: "
        print self.classes

        for index, _ in enumerate(self.classes):
            print "\nClass: ", _
            print "----------------------\nPrecision:"
            print self.precision(index)

            print "----------------------\nRecall"
            print self.recall(index)

            print "----------------------\nF Measure"
            print self.f_measure(index)

    def evaluate_problems(self):
        self.calculate_confusion_matrices()

        self.calculate_true_positive()
        self.calculate_false_positive()
        self.calculate_false_negative()
        # the other 3 are used to calculate the true negative, so this must be done last
        self.calculate_true_negative()

    def calculate_confusion_matrices(self):
        self.confusion_matrices = confusion_matrix(self.actual_results, self.predicted_results)

    def calculate_true_positive(self):
        # make a tuple with elements for each class in the problem
        self.true_positive = [None] * 2

        # calculate the number of true positives for the class
        for index in range(0,2):
            self.true_positive[index] = self.confusion_matrices[index,index]

    def calculate_false_positive(self):
        # make a tuple with elements for each class in the problem
        self.false_positive =[None] * 2

        # calculate the number of true positives for the class
        for index in range(0,2):
            self.false_positive[index] = np.sum(self.confusion_matrices[:, index]) - np.asscalar(self.confusion_matrices[index,index])

    def calculate_false_negative(self):
        # make a tuple with elements for each class in the problem
        self.false_negative = [None] * 2

        # calculate the number of true positives for the class
        for index in range(0,2):
            self.false_negative[index] = np.sum(self.confusion_matrices[index,:]) - np.asscalar(self.confusion_matrices[index,index])

    def calculate_true_negative(self):
        # make a tuple with elements for each class in the problem
        self.true_negative = [None] * 2

        total_count = np.sum(self.confusion_matrices)
        # calculate the number of true positives for the class
        for index in range(0,2):
            temp = total_count - (self.true_positive[index])
            temp -= self.false_positive[index]
            temp -= self.false_negative[index]
            self.true_negative[index] = temp

    # exists by summing up all true positives by the total number of test samples
    def accuracy(self):
        total_count = np.sum(self.confusion_matrices)
        return np.sum(self.true_positive)*1.0 / total_count

    # returns a tuple containing recalls for each class
    def recall(self, class_index):
        return self.true_positive[class_index]*1.0/(self.true_positive[class_index]+ self.false_negative[class_index])

    def precision(self, class_index):
        return self.true_positive[class_index]*1.0 /(self.true_positive[class_index] + self.false_positive[class_index])

    def f_measure(self, class_index):
        return 2.0/(1.0/ self.precision(class_index) + 1.0/self.recall(class_index))


# splits up data for crossfolds
class Crossfold:
    def __init__(self, data, num_times):
        self.data = data
        self.problem_data = None
        self.problem_test_data = None
        self.num_kfolds = None
        self.prepare_kfold(num_times=num_times)

    # assumes data has already been mixed, otherwise issues may arise when doing classification
    def prepare_kfold(self, num_times):
        self.num_kfolds = num_times
        total_number_examples = np.size(self.data, 0)
        num_examples = int(total_number_examples*1.0 / self.num_kfolds)
        iterator = 0

        # set up tuples to hold the data as its split up for the kfold testing
        self.problem_data = [None]*num_times
        self.problem_test_data = [None]*num_times

        for i in range(0, num_times):
            # position of end of end of the test data
            iterator_end = iterator + num_examples

            # save the test data
            self.problem_test_data[i] = self.data[iterator:iterator_end]
            # merge two numpy arrays, 0 to the start of the test data, and end of of the test data to the end
            if iterator != 0 and iterator_end != total_number_examples:
                self.problem_data[i] = np.concatenate((self.data[:iterator], self.data[iterator_end:]), axis=0)
            else:
                if iterator == 0:
                    self.problem_data[i] = self.data[iterator_end:]
                else:
                    self.problem_data[i] = self.data[:iterator]

            # increment iterator to the next position
            iterator += num_examples

    def get_problem_data(self):
        return self.problem_data

    def get_problem_test_data(self):
        return self.problem_test_data



