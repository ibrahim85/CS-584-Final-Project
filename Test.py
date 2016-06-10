import DataPrep
import GeneticAlgo
import Evaluation
import numpy as np

Num_trials = 2
num_stocks = 2

test = DataPrep.DataFromFile()
data = test.create_params(num_stocks)
class_data = test.get_class()

test = GeneticAlgo.GeneticAlgo(data, class_data)
my_svm = test.eval(Num_trials)
