import numpy as np
from IPython.display import clear_output

parameters = {
    'Neq': 3
    ,'a1': 1
    ,'a2': 2
    ,'GP': 0.5
    ,'GCP_TYPE': 1
    , 'PARTITION_TYPE': 1
    , 'MUTATION_PROB': 0.3
}


class Particle(object):
    def __init__(self, vector):
        self.vector = vector
        self.score = None

    def real2int(self):  # for discrete problems only
        return np.apply_along_axis(np.round, axis=0, arr=self.vector)

    def __str__(self):
        return f'Vector:{self.vector} & Score:{self.score}'


class EO(object):
    def __init__(self, problem, popSize, maxIter, parameters, searchSpace='discrete'):

        self.params = parameters

        self.initializer = problem.initializer
        self.objectiveFunc = problem.objectiveFunc
        self.vecBound = problem.vecBound

        self.popSize = popSize
        self.maxIter = maxIter
        self.searchSpace = searchSpace

        self.dim = None
        self.C = []
        self.Ceq = []
        self.equal_range_thresholds = None

    def run(self):
        self.initialize()
        for i in range(self.maxIter):
            #print(f'-----------starting iteration {i + 1}-----------')
            clear_output(wait=True)

            self.getScore()
            self.updateCeq()
            #self.eoEvolve(i + 1)
            self.eoEvolve_Modify(i + 1)

    def get_best_x(self):
        return self.Ceq[0].vector

    def initialize(self):
        #print(f'-----------initializing Population-----------')
        self.C = [Particle(self.initializer()) for each in range(self.popSize)]
        self.dim = len(self.C[0].vector)
        self.equal_range_thresholds = self.hist_uniform_division()

    def getScore(self):
        for each in self.C:
            self.objectiveFunc(each)

    def updateCeq(self):
        dummy = Particle([]);
        dummy.score = float('-inf')

        cs = sorted(self.C, key=lambda x: x.score, reverse=True);
        cs.append(dummy)
        ceqs = sorted(self.Ceq, key=lambda x: x.score, reverse=True);
        ceqs.append(dummy)

        from copy import copy
        Ceq, i, j = [], 0, 0
        for each in range(self.params['Neq'] - 1):
            if cs[i].score > ceqs[j].score:
                Ceq.append(copy(cs[i]))
                i += 1
            else:
                Ceq.append(copy(ceqs[j]))
                j += 1

        Ceqavg = Particle(np.average([each.vector for each in Ceq], axis=0))
        self.objectiveFunc(Ceqavg)
        Ceq.append(Ceqavg)

        if self.searchSpace == 'discrete':
            for i, each in enumerate(Ceq):
                Ceq[i].vector = each.real2int()

        self.Ceq = Ceq

    def eoEvolve_Modify(self, itr):
        mutation_prob = self.params['MUTATION_PROB']
        pop_partition = np.round(np.random.uniform(0, 1, self.popSize), 0)
        i = 0
        for each_C in self.C:
            is_mutation = np.random.rand()
            if pop_partition[i] == 0:
                if is_mutation >= mutation_prob:
                    self.mutation2(each_C)
                else:
                    self.eoEvolveAlg(itr, each_C, is_last_coef=True)
            else:
                if is_mutation >= mutation_prob:
                    self.mutation1(each_C)
                else:
                    self.eoEvolveAlg(itr, each_C, is_last_coef=False)
            thresholds, is_update = self.repair(each_C)
            if is_update:
                each_C.vector = thresholds

    def mutation1(self, each_C):
        is_mutation = np.random.rand()
        if is_mutation > 0.5:
            nTh = self.dim
            each_C.vector = np.int_(each_C.vector + np.sign(np.random.random(nTh)-0.5)*np.random.randint(1, nTh)) % 256


    def mutation2(self, each_C):
        modified_thresholds, modify_count = self.modify_thresholds_equal_range_distribution(each_C.vector, self.equal_range_thresholds)
        if modify_count == 0:
            self.mutation1(each_C)
        else:
            each_C.vector = modified_thresholds

    def repair(self, each_C):
        nTh = self.dim
        thresholds = np.int_(np.sort(each_C.vector))
        unique_thresholds = np.unique(thresholds)
        unique_thresholds_num = unique_thresholds.size
        if unique_thresholds_num < nTh:
            thresholds_diff = np.diff(thresholds)
            repair_range = self.get_repair_range(unique_thresholds, unique_thresholds_num)
            repair_range_idx_l = unique_thresholds_num
            repair_range_idx_h = 255
            i = 0
            while i < nTh-1:
                if thresholds_diff[i] == 0:
                    update_idx = np.random.randint(repair_range_idx_l, repair_range_idx_h)
                    update_value = repair_range[update_idx]
                    thresholds[i] = update_value
                    repair_range[update_idx] = repair_range[repair_range_idx_l]
                    repair_range_idx_l += 1
                i += 1
        is_update = unique_thresholds_num < nTh
        return thresholds, is_update


    def get_repair_range(self, unique_thresholds, unique_thresholds_num):
        unique_thresholds_sorted = np.sort(unique_thresholds)
        repair_range = np.array(range(256))
        i = 0
        while i < unique_thresholds_num:
            if unique_thresholds_sorted[i] > 255:
                unique_thresholds_sorted[i] = 255
            repair_range[np.int_(unique_thresholds_sorted[i])] = -1
            i += 1
        repair_range = np.sort(repair_range)
        return repair_range




   # def hist_uniform_division(self):
   #     nTh = self.dim
   #     partition_range = 255/(nTh+1)
   #     equal_range_thresholds = np.zeros(nTh+2)
   #     i = 1
   #     while i < nTh:
   #         equal_range_thresholds[i] = partition_range + i*nTh
   #         i += 1
   #     equal_range_thresholds[i] = 255
   #     return equal_range_thresholds
    def hist_uniform_division(self):
        nTh = self.dim
        partition_range = 256 / (nTh + 1)
        equal_range_thresholds = np.zeros(nTh + 2)
        for i in range(1, nTh + 1):
            equal_range_thresholds[i] = i * partition_range
        equal_range_thresholds[nTh + 1] = 255
        return np.round(equal_range_thresholds, 0)

    def modify_thresholds_equal_range_distribution(self, thresholds, equal_range_thresholds):
        nTh = self.dim
        thresholds_sorted = np.sort(thresholds)
        #out_of_range_idx_list = []
        i = 0
        modify_count = 0
        while i < nTh:
            if thresholds_sorted[i] < equal_range_thresholds[i] or thresholds_sorted[i] > equal_range_thresholds[i+1]:
                thresholds_sorted[i] = np.float_(np.random.randint(equal_range_thresholds[i], equal_range_thresholds[i+1]))
                modify_count += 1
            i += 1
        return thresholds_sorted, modify_count


    def eoEvolveAlg(self, itr, each_C, is_last_coef):
        a1 = self.params['a1']
        a2 = self.params['a2']
        GP = self.params['GP']
        Neq = self.params['Neq']
        GCP_TYPE = self.params['GCP_TYPE']

        dim = self.dim
        maxIter = self.maxIter
        vecBound = self.vecBound

        r = np.random.rand(dim)
        lamda = np.random.rand(dim)

        t = (1 - itr / maxIter) ** (a2 * itr / maxIter)
        F = a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1)

        GCP = self.cal_GCP(GP, GCP_TYPE, dim)
        GCP = np.ones(dim) * GCP
        j = np.random.randint(0, Neq)
        C = each_C.vector
        Ceq = np.random.choice(self.Ceq).vector
        G0 = GCP * Ceq - lamda * C
        G = G0 * F
        newC = Ceq + (C - Ceq) * F
        if is_last_coef:
            newC += G * (1 - F) / lamda
        newC = np.array([min(max(i, j), k) for i, j, k in zip(vecBound[0], newC, vecBound[1])])
        each_C.vector = newC

    def eoEvolve(self, itr):
        a1 = self.params['a1']
        a2 = self.params['a2']
        GP = self.params['GP']
        Neq = self.params['Neq']
        GCP_TYPE = self.params['GCP_TYPE']
        PARTITION_TYPE = self.params['PARTITION_TYPE']

        dim = self.dim
        maxIter = self.maxIter
        vecBound = self.vecBound

        r = np.random.rand(dim)
        #r1 = np.random.rand()
        #r2 = np.random.rand()
        lamda = np.random.rand(dim)

        t = (1 - itr / maxIter) ** (a2 * itr / maxIter)
        F = a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1)

        #if r2 >= GP:
            #GCP = 0.5 * r1
        #else:
            #GCP = 0

        #GCP = 0.5*np.random.randint(1, dim)
        GCP = self.cal_GCP(GP, GCP_TYPE, dim)
        GCP = np.ones(dim) * GCP
        j = np.random.randint(0, Neq)

        if PARTITION_TYPE == 0:
            pop_partition = np.zeros(self.popSize)
        else:
            pop_partition = np.round(np.random.uniform(0, 1, self.popSize), 0)
        i = 0
        for each_C in self.C:
            C = each_C.vector
            Ceq = np.random.choice(self.Ceq).vector

            G0 = GCP * Ceq - lamda * C
            G = G0 * F

            newC = Ceq + (C - Ceq) * F #+ G * (1 - F) / lamda
            if pop_partition[i] == 0:
                newC += G * (1 - F) / lamda
            newC = np.array([min(max(i, j), k) for i, j, k in zip(vecBound[0], newC, vecBound[1])])
            each_C.vector = newC
            i += 1


    def cal_GCP(self, GP, GCP_TYPE, dim):
        if GCP_TYPE == 0:
            r1 = np.random.rand()
            r2 = np.random.rand()
            if r2 >= GP:
                GCP = 0.5 * r1
            else:
                GCP = 0
        else:
            GCP = 0.5 * np.random.randint(1, dim)
        return GCP

