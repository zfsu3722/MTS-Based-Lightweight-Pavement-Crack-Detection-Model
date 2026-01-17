import numpy as np
from sko.PSO import PSO


def threshold_sequence_constraint(x):
    x_len = len(x)
    i = 1
    ret = -1
    ret_logic = True
    while i < x_len:
        ret_logic = ret_logic and x[i] > x[i-1]
        if not ret_logic:
            ret = 1
            break
        i += 1
    return ret

#super().__init__(func, n_dim, pop, max_iter, lb, ub, w, c1, c2, constraint_eq, constraint_ueq, verbose, dim, n_processes)

    #def update_X(self):
        #self.X = self.X + self.V
        #lsp_switch = np.random.random()
        #if self.lsp_on and lsp_switch > self.lsp_prob:
            #lsp_coef = np.random.random()
            #self.X = self.X + self.V*lsp_coef
        #self.X = np.clip(self.X, self.lb, self.ub)


class PSOLsp(PSO):
    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=0.5, c2=0.5,
                 constraint_eq=tuple(), constraint_ueq=tuple(), verbose=False
                 , dim=None, lsp_on=True, lsp_prob=0.7):
        super().__init__(func, n_dim, pop, max_iter, lb, ub, w, c1, c2, constraint_eq, constraint_ueq, verbose, dim)
        self.is_lsp_on = lsp_on
        self.lsp_prob = lsp_prob


    def get_best_x(self):
        return self.gbest_x

    def update_X(self):
        self.X = self.X + self.V
        i = 0
        if self.is_lsp_on:
            while i < self.pop:
                lsp_switch = np.random.random()
                if lsp_switch < self.lsp_prob:
                    lsp_coef = np.random.random(self.n_dim)
                    lsp_switch = np.random.random()
                    if lsp_switch > self.lsp_prob:
                        mv = 1
                    else:
                        mv = 0
                    self.X[i, :] = self.X[i, :] + self.ub * lsp_coef * mv
                else:
                    r3 = np.random.random()
                    self.lsp_prob = (1-r3)*self.lsp_prob+r3
                    p = np.random.randint(1, self.pop)
                    q = np.random.randint(1, self.pop)
                    self.X[i, :] = self.X[i, :] + (self.X[p-1, :] - self.X[q-1, :])*self.lsp_prob
                i += 1
        self.X = np.clip(self.X, self.lb, self.ub)


class SegObjFunBase:
    def __init__(self, img, dim):
        self.hist_raw = np.histogram(img, bins=range(257))
        self.hist = np.float_(self.hist_raw[0])
        self.hist_bins = np.float_(self.hist_raw[1])
        self.hist_bins = np.delete(self.hist_bins, 256)
        self.c_hist = np.cumsum(self.hist)
        self.cdf = np.cumsum(self.hist_bins * self.hist)
        #self.c_hist = np.append(self.c_hist, [0])
        #self.cdf = np.append(self.cdf, [0])
        self.dim = dim
        self.vecBound = (np.zeros(dim), np.ones(dim)*255)
        hist_bins_inc = self.hist_bins + 1
        self.log_df_total = np.sum(hist_bins_inc * self.hist * np.log(hist_bins_inc))

    def initializer(self):
        vector_list = [np.random.uniform(i, j) for i, j in zip(*self.vecBound)]
        return np.round(np.array(vector_list), 0)

    def objectiveFunc(self, particle):
        thresholds = particle.vector
        particle.score = self.seg_obj(thresholds)

    def seg_obj(self, thresholds):
        return 0

    @staticmethod
    def obj_info(self):
        print("Base Obj for Otsu and Kapur")


class OtsuObjFun(SegObjFunBase):
    def __init__(self, img, dim=0):
        super().__init__(img, dim)

        #self.hist = np.float_(np.histogram(img, bins=range(256))[0])
        #self.c_hist = np.cumsum(self.hist)
        #self.cdf = np.cumsum(np.arange(len(self.hist)) * self.hist)
        #self.c_hist = np.append(self.c_hist, [0])
        #self.cdf = np.append(self.cdf, [0])

    def seg_obj(self, thresholds):
        e_thresholds = [0]
        e_thresholds.extend(np.sort(np.round(thresholds, 0).astype(np.uint8)))
        e_thresholds.extend([len(self.hist) - 1])
        v_obj = self.cal_otsu_obj(e_thresholds)
        return -v_obj

    def cal_otsu_obj(self, thresholds):
        variance = 0
        for i in range(len(thresholds) - 1):
            # Thresholds
            t1 = thresholds[i] #+ 1
            t2 = thresholds[i + 1]
            # Cumulative histogram
            #if t1 != 0:
            weight = self.c_hist[t2] - self.c_hist[t1] #t-1
            #else:
                #weight = 0
            # Region CDF
            r_cdf = self.cdf[t2] - self.cdf[t1] #t-1
            # Region mean
            r_mean = r_cdf / weight if weight != 0 else 0
            variance += weight * r_mean ** 2
        return variance


class KapurObjFun(SegObjFunBase):
    def __init__(self, img, dim=0):
        super().__init__(img, dim)

    def seg_obj(self, thresholds):
        e_thresholds = [0]
        e_thresholds.extend(np.sort(np.round(thresholds, 0).astype(np.uint8)))
        e_thresholds.extend([len(self.hist) - 1])
        v_obj = self.cal_kapur_obj(e_thresholds)
        return -v_obj

    def cal_kapur_obj(self, thresholds):
        total_entropy = 0
        for i in range(len(thresholds) - 1):
            # Thresholds
            t1 = thresholds[i] #+ 1
            t2 = thresholds[i + 1]

            # print(thresholds, t1, t2)

            # Cumulative histogram
            hc_val = self.c_hist[t2] - self.c_hist[t1]

            # Normalized histogram
            h_val = self.hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1

            # entropy
            entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()

            # Updating total entropy
            total_entropy += entropy
        return total_entropy


class AGDivergence(SegObjFunBase):
    def __init__(self, img, dim=0):
        super().__init__(img, dim)
        self.h_min = 1e-5

    def seg_obj(self, thresholds):
        e_thresholds = [0]
        e_thresholds.extend(np.sort(np.round(thresholds, 0).astype(np.uint8)))
        e_thresholds.extend([len(self.hist) - 1])
        agd_measure_total = 0
        for i in range(len(e_thresholds) - 1):
            t_low = e_thresholds[i]
            t_high = e_thresholds[i+1]
            m, i_m_sum_coef_1, i_m_sum_coef_2, i_m_sum_coef_3 = self.cal_seg_coef(t_low, t_high)
            if m > 0:
                agd_measure = self.cal_agd_measure(i_m_sum_coef_1, i_m_sum_coef_2, i_m_sum_coef_3)
                agd_measure_total += agd_measure
        return -agd_measure_total

    def cal_seg_coef(self, t_low, t_high):
        weighted_sum = self.cdf[t_high] - self.cdf[t_low]
        count_sum = self.c_hist[t_high] - self.c_hist[t_low]
        if count_sum > 0 and weighted_sum > 0:
            if t_low == -1:
                t_low = 0
            m = weighted_sum/count_sum
            i_m_sum_coef_b = (self.hist_bins[t_low:t_high+1] + m)/2
            i_m_sum_coef_1 = i_m_sum_coef_b/m
            i_m_sum_coef_2 = i_m_sum_coef_b * self.hist[t_low:t_high+1]
            seg_bins = self.hist_bins[t_low:t_high+1]
            if seg_bins[0] == 0:
                seg_bins[0] = self.h_min
            i_m_sum_coef_3 = i_m_sum_coef_b/seg_bins
        else:
            m = 0
            i_m_sum_coef_1 = None
            i_m_sum_coef_2 = None
            i_m_sum_coef_3 = None
        return m, i_m_sum_coef_1, i_m_sum_coef_2, i_m_sum_coef_3

    @staticmethod
    def cal_agd_measure(i_m_sum_coef_1, i_m_sum_coef_2, i_m_sum_coef_3):
        agd_measure_1 = np.sum(i_m_sum_coef_2*np.log(i_m_sum_coef_3))
        agd_measure_2 = np.sum(i_m_sum_coef_2 * np.log(i_m_sum_coef_1))
        agd_measure = agd_measure_1 + agd_measure_2
        return agd_measure


class CrossEntropyObjFun(SegObjFunBase):
    def __init__(self, img, dim=0):
        super().__init__(img, dim)
        self.h_min = 1e-5

    def seg_obj(self, thresholds):
        e_thresholds = [0]
        e_thresholds.extend(np.sort(np.round(thresholds, 0).astype(np.uint8)))
        e_thresholds.extend([len(self.hist) - 1])
        cross_entropy_measure_seg_total = 0
        for i in range(len(e_thresholds) - 1):
            t_low = e_thresholds[i]
            t_high = e_thresholds[i + 1]
            cdf_seg = self.cdf[t_high] - self.cdf[t_low]
            c_hist_seg = self.c_hist[t_high] - self.c_hist[t_low]
            if c_hist_seg > 0 and cdf_seg > 0:
                m_seg = cdf_seg/c_hist_seg
            else:
                m_seg = self.h_min
            cross_entropy_measure_seg = np.sum(self.hist_bins[t_low:t_high+1]*self.hist[t_low:t_high+1]*np.log(m_seg))
            cross_entropy_measure_seg_total += cross_entropy_measure_seg
        cross_entropy_measure = np.abs(self.log_df_total - cross_entropy_measure_seg_total)
        return -cross_entropy_measure
