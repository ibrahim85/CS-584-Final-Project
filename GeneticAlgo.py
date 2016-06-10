import numpy as np
import random
import SVM
import DataPrep


class GeneticAlgo:

    def __init__(self, x, y, num_chromosomes=3, chromosome_mutation_rate=.5):
        self.x = np.array(x)
        self.y = y
        self.param_count = np.shape(self.x)[1]
        self.num_chromosomes = num_chromosomes
        self.chromosome_svm = [None]*self.num_chromosomes
        self.chromosomes = np.zeros((self.num_chromosomes, self.param_count))
        self.chromosome_mutation_rate = chromosome_mutation_rate
        self.chromosome_data = [None]*self.num_chromosomes
        self.chromosome_accuracy = [0.0]*self.num_chromosomes

        # set up chromosomes
        # 0 - every even gene index
        # 1 - every odd gene index
        # 2 - every even gene index and indexes divisible by 3
        # 3 - every even gene index and indexes divisible by 5 and 7
        for i in range(0,self.param_count):
            if i%2 == 0:
                self.chromosomes[0][i] = 1
                self.chromosomes[2][i] = 1
                #self.chromosomes[3][i] = 1
            if (i+1) % 2 == 0:
                self.chromosomes[1][i] = 1
            if i % 3 == 0:
                self.chromosomes[2][i] = 1
            #if i % 5 == 0 or i % 7 == 0:
                #self.chromosomes[3][i] = 1

    def create_data_from_chromosome(self):

        for index, c in enumerate(self.chromosomes):
            # get a list of all non zero values. The corresponding parameter will be tested
            params = np.nonzero(np.array(c))[0]

            self.chromosome_data[index] = self.x[:, params]
        return self.chromosome_data

    # changes a gene's state in the chromosome
    def mutate_chromosomes(self, chromosome):
        # random number of mutations to occur
        num_to_mutate = int(random.randint(0,self.param_count-1) * self.chromosome_mutation_rate)

        for i in range(0,num_to_mutate):
            # select a gene to reverse, so 0->1 and 1->0. Genes with a value of 1 will be used in the SVM
            pos = random.randint(0, self.param_count-1)
            if chromosome[pos] == 0:
                chromosome[pos] = 1
            else:
                chromosome[pos] = 0
        return chromosome

    # combines two chromosomes and mutates the result
    def crossover(self, mom, dad):
        # select the position to cut off parent chromosomes for recombination
        crossover_pos = int(random.randint(1, self.param_count-1))
        # create child chromosome by combining parts from parents
        moms_part = mom[: crossover_pos]
        dads_part = dad[crossover_pos:]
        child = np.append(moms_part, dads_part)

        # mutate genes in child
        child = self.mutate_chromosomes(child)
        return child

    def test_chromosomes(self):
        self.create_data_from_chromosome()
        for index, chromosome in enumerate(self.chromosome_data):
            print "Testing  chromosome: ", index
            svm = SVM.SvmFreeParam(chromosome, self.y)
            # increase accuracy so we can select chromosomes easier
            self.chromosome_accuracy[index] = svm.optimize_params()*1000
            self.chromosome_svm[index] = svm


    def create_next_generation(self):
        temp_chromosomes = [None]*self.num_chromosomes
        max_val = np.sum(self.chromosome_accuracy)
        tiers = [None]*self.num_chromosomes

        tiers[0] = self.chromosome_accuracy[0]
        tiers[1] = self.chromosome_accuracy[1] + tiers[0]
        tiers[2] = self.chromosome_accuracy[2] + tiers[1]
        #tiers[3] = self.num_chromosomes[3] + tiers[2]

        for index, chromosome in enumerate(self.chromosomes):
            pos = random.uniform(0, max_val)
            i = 0
            while pos > tiers[i]:
                i += 1

            pos = random.uniform(0, max_val)
            j = 0
            while pos > tiers[j]:
                j += 1

            temp_chromosomes[index] = self.crossover(self.chromosomes[i], self.chromosomes[j])

    def eval(self, num_evolutions):
        for i in range(0, num_evolutions):
            print "Evolution: ", i
            self.test_chromosomes()
            self.create_next_generation()

        self.test_chromosomes()
        max_accuracy_index = self.chromosome_accuracy.index(max(self.chromosome_accuracy))
        print "Max accuracy", max(self.chromosome_accuracy)/1000

        return self.chromosome_svm[max_accuracy_index]

