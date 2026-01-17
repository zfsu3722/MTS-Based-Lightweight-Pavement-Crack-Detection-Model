import random

import numpy as np


class HBO(object):
    def __init__(self, searchAgents, Max_iter, lb, ub, dim, fobj, cycles, degree):
        self.searchAgents = searchAgents
        self.Max_iter = Max_iter
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.fobj = fobj
        #self.Positions = None
        self.cycles = cycles
        self.degree = degree
        self.best_solution = None
        #self.initialize()

    def positions_initialize(self):
        Boundary_no = self.ub.size
        if Boundary_no == 1:
            Positions = np.random.random((self.searchAgents, self.dim))*(self.ub-self.lb)+self.lb
            #Counter_Positions = -(Positions - self.lb - self.ub)
        else:
            i = 0
            Positions = np.zeros((self.searchAgents, self.dim))
            Counter_Positions = np.zeros((self.searchAgents, self.dim))
            while i < self.dim:
                ub_i = self.ub[i]
                lb_i = self.lb[i]
                Positions[:, i] = np.random.random(self.searchAgents)*(ub_i - lb_i)+lb_i
                #Counter_Positions[:, i] = -(Positions[:, i] - ub_i - lb_i)
                i += 1
        return Positions#, Counter_Positions

    def cal_fitness(self, Solutions, solution_idx):
        Counter_Solutions = (self.ub + self.lb) - Solutions[solution_idx, :]
        fitness = self.fobj(Solutions[solution_idx, :])
        counter_fitness = self.fobj(Counter_Solutions)
        if counter_fitness < fitness:
            Solutions[solution_idx, :] = Counter_Solutions
            fitness = counter_fitness
        return fitness

    def heap_initialize(self):
        treeHeight = np.ceil((np.log10(self.searchAgents * self.degree - self.searchAgents + 1) / np.log10(self.degree)))
        fevals = 0
        Leader_pos = np.zeros(self.dim)
        Solutions = self.positions_initialize()
        Leader_score = np.inf
        fitnessHeap = np.zeros((self.searchAgents, 2)) + np.inf
        #build heap here
        c = 0
        while c < self.searchAgents:
            fitness = self.cal_fitness(Solutions, c)#self.fobj(Solutions[c, :])
            fitnessHeap[c, 0] = fitness
            fitnessHeap[c, 1] = np.int_(c)
            t = c
            while t > 0:
                parentInd = np.int_(np.floor((t + 1) / self.degree))
                if fitnessHeap[t, 0] >= fitnessHeap[parentInd, 0]:
                    break
                else:
                    tempFitness = fitnessHeap[t, :]
                    fitnessHeap[t, :] = fitnessHeap[parentInd, :]
                    fitnessHeap[parentInd, :] = tempFitness
                t = parentInd
            if fitness <= Leader_score:
                Leader_score = fitness
                Leader_pos = Solutions[c, :]
            c += 1
        Convergence_curve = np.zeros(self.Max_iter+1)
        return Solutions, Leader_score, Leader_pos, fitnessHeap, fevals, treeHeight, Convergence_curve

    def colleaguesLimitsGenerator(self):
        colleaguesLimits = np.zeros((self.searchAgents, 2))
        c = self.searchAgents
        while c > 0:
            hi = np.ceil((np.log10(c * self.degree - c + 1) / np.log10(self.degree))) - 1
            lowerLim = ((self.degree * np.power(self.degree, (hi - 1)) - 1) / (self.degree - 1) + 1)
            upperLim = (self.degree * np.power(self.degree, hi) - 1) / (self.degree - 1)
            colleaguesLimits[c-1, 0] = lowerLim
            colleaguesLimits[c-1, 1] = upperLim
            c -= 1
        return colleaguesLimits

    def run(self):
        Solutions, Leader_score, Leader_pos, fitnessHeap, fevals, treeHeight, Convergence_curve = self.heap_initialize()
        colleaguesLimits = self.colleaguesLimitsGenerator()
        #Convergence_curve = np.zeros(self.Max_iter)
        itPerCycle = np.int_(np.round(self.Max_iter / self.cycles, 0))
        qtrCycle = np.int_(np.round(itPerCycle / 4, 0))
        it = 0
        while it < self.Max_iter:
            gamma = (np.mod(it, itPerCycle) + 1) / qtrCycle
            gamma = np.abs(2 - gamma)
            c = np.int_(self.searchAgents - 1)
            while c > 0:
                if c == 0:
                    continue
                else:
                    parentInd = np.int_(np.floor(c+1) / self.degree)
                    curSol = Solutions[np.int_(fitnessHeap[c, 1]), :]
                    parentSol = Solutions[np.int_(fitnessHeap[parentInd, 1]), :]
                    if colleaguesLimits[c, 1] > self.searchAgents:
                        colleaguesLimits[c, 1] = self.searchAgents
                    colleagueInd = c
                    while colleagueInd == c:
                        colleagueInd = np.random.randint(colleaguesLimits[c, 0], colleaguesLimits[c, 1])
                    colleagueSol = Solutions[np.int_(fitnessHeap[colleagueInd, 1]), :]

                    j = 0
                    while j < self.dim:
                        p1 = (1 - (it+1) / (self.Max_iter))
                        p2 = p1 + (1 - p1) / 2
                        r = np.random.rand()
                        rn = (2 * np.random.rand() - 1)

                        if r < p1:
                            continue
                        elif r < p2:
                            D = np.abs(parentSol[j] - curSol[j])
                            curSol[j] = parentSol[j] + rn * gamma * D
                        else:
                            if fitnessHeap[colleagueInd, 0] < fitnessHeap[c, 0]:
                                D = np.abs(colleagueSol[j] - curSol[j])
                                curSol[j] = colleagueSol[j] + rn * gamma * D
                            else:
                                D = np.abs(colleagueSol[j] - curSol[j])
                                curSol[j] = curSol[j] + rn * gamma * D
                        j += 1
                Flag4ub = np.where(curSol > self.ub, 1, 0)
                Flag4lb = np.where(curSol < self.lb, 1, 0)
                NFlag4ub = np.where(curSol > self.ub, 0, 1)
                NFlag4lb = np.where(curSol < self.lb, 0, 1)
                curSol = (curSol * (NFlag4ub * NFlag4lb)) + self.ub * Flag4ub + self.lb * Flag4lb
                newFitness = self.fobj(curSol)
                fevals = fevals + 1
                if newFitness < fitnessHeap[c, 0]:
                    fitnessHeap[c, 0] = newFitness
                    Solutions[np.int_(fitnessHeap[c, 1]), :] = curSol
                    newFitness = self.cal_fitness(Solutions, np.int_(fitnessHeap[c, 1]))
                if newFitness < Leader_score:
                    Leader_score = newFitness
                    Leader_pos = Solutions[np.int_(fitnessHeap[c, 1]), :]
                t = c
                while t > 0:
                    parentInd = np.int_(np.floor((t + 1) / self.degree))
                    if fitnessHeap[t, 0] >= fitnessHeap[parentInd, 0]:
                        break
                    else:
                        tempFitness = fitnessHeap[t, :]
                        fitnessHeap[t, :] = fitnessHeap[parentInd, :]
                        fitnessHeap[parentInd, :] = tempFitness
                    t = parentInd
                c -= 1
            Convergence_curve[it+1] = Leader_score
            it += 1
        #Best_solution = Solutions[np.int_(Leader_pos), :]
        self.best_solution = Leader_pos
        return Leader_pos, Leader_score, Convergence_curve

    def get_best_x(self):
        return self.best_solution


class SingleRandomSearch(object):
    def __init__(self, searchAgents, lb, ub, dim, fobj):
        self.searchAgents = searchAgents
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.fobj = fobj

    def positions_initialize(self):
        Boundary_no = self.ub.size
        if Boundary_no == 1:
            Positions = np.random.random((self.searchAgents, self.dim))*(self.ub-self.lb)+self.lb
            #Counter_Positions = -(Positions - self.lb - self.ub)
        else:
            i = 0
            Positions = np.zeros((self.searchAgents, self.dim))
            Counter_Positions = np.zeros((self.searchAgents, self.dim))
            while i < self.dim:
                ub_i = self.ub[i]
                lb_i = self.lb[i]
                Positions[:, i] = np.random.random(self.searchAgents)*(ub_i - lb_i)+lb_i
                #Counter_Positions[:, i] = -(Positions[:, i] - ub_i - lb_i)
                i += 1
        return Positions#, Counter_Positions

    def cal_fitness(self, Solutions, solution_idx, is_counter=True):
        Counter_Solutions = (self.ub + self.lb) - Solutions[solution_idx, :]
        fitness = self.fobj(Solutions[solution_idx, :])
        counter_fitness = self.fobj(Counter_Solutions)
        if is_counter and counter_fitness < fitness:
            Solutions[solution_idx, :] = Counter_Solutions
            fitness = counter_fitness
        return fitness

    def run(self):
        Solutions = self.positions_initialize()
        i = 0
        is_counter = True
        score_list = np.ones(self.searchAgents)
        opt_score = np.inf
        opt_solution = None
        while i < self.searchAgents:
            fitness = self.cal_fitness(Solutions, i, is_counter)
            score_list[i] = fitness
            if fitness < opt_score:
                opt_score = fitness
                opt_solution = Solutions[i, :]
            i += 1
        return opt_score, opt_solution, score_list
