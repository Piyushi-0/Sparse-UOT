import torch
import torch.nn.functional as F
import numpy as np
import ot
from util import *
from mparts import *
from sparse_ot.sparse_repr_autok import get_gamma
from sparse_ot.utils import postprocess_gamma
from sparse_ot.utils import get_G, get_dist

def BERTScore_Alignment(vecX, vecY, merge_type):
    vecX = F.normalize(vecX)
    vecY = F.normalize(vecY)
    matrix = torch.matmul(vecX, vecY.t())
    alignments_x = set(['{0}-{1}'.format(i, torch.argmax(matrix[i])) for i in range(matrix.shape[0])])
    alignments_y = set(['{0}-{1}'.format(torch.argmax(matrix[:, i]), i) for i in range(matrix.shape[1])])

    if merge_type == 'intersect':
        alignments = alignments_x & alignments_y
    elif merge_type == 'union':
        alignments = alignments_x | alignments_y
    elif merge_type == 'grow_diag':
        alignments = grow_diag(alignments_x, alignments_y, vecX.shape[0], vecY.shape[0])
    elif merge_type == 'grow_diag_final':
        alignments = grow_diag_final(alignments_x, alignments_y, vecX.shape[0], vecY.shape[0])

    return alignments


class Aligner:
    def __init__(self, ot_type, chimera, dist_type, weight_type, distortion, thresh, outdir, **kwargs):
        self.ot_type = ot_type
        self.chimera = chimera
        self.dist_type = dist_type
        self.weight_type = weight_type
        self.distotion = distortion
        self.thresh = thresh
        self.lda = kwargs['lda']
        self.lda3 = kwargs['lda3']
        self.khp = kwargs['khp']
        self.ktype = kwargs['ktype']
        self.max_itr = kwargs['max_itr']
        self.ws = kwargs['ws']
        self.K = kwargs['K']
        self.s = kwargs['s']
        self.all_gamma = kwargs['all_gamma']
        self.keuc = kwargs['keuc']
        self.outdir = outdir
        self.logfile = kwargs['logfile']

        self.dist_func = compute_distance_matrix_cosine if dist_type == 'cos' else compute_distance_matrix_l2
        if weight_type == 'uniform':
            self.weight_func = compute_weights_uniform
        else:
            self.weight_func = compute_weights_norm

    def compute_alignment_matrices(self, s1_vecs, s2_vecs):
        self.align_matrices = []
        for vecX, vecY in zip(s1_vecs, s2_vecs):
            P = self.compute_optimal_transport(vecX, vecY)
            if torch.is_tensor(P):
                P = P.to('cpu').numpy()

                self.align_matrices.append(P)
            else: #NOTE: not needed now
                K = len(P) # how many OT plans got from all_gamma
                self.align.matrices.append(np.split(torch.cat(P).cpu().numpy(), K))

    def get_alignments(self, thresh, assign_cost=False):

        self.thresh = thresh
        all_alignments = []
        for P in self.align_matrices:
            alignments = self.matrix_to_alignments(P, assign_cost)
            all_alignments.append(alignments)

        return all_alignments

    def matrix_to_alignments(self, P, assign_cost):
        alignments = set()
        align_pairs = np.transpose(np.nonzero(P > self.thresh))
        if assign_cost:
            for i_j in align_pairs:
                alignments.add('{0}-{1}-{2:.4f}'.format(i_j[0], i_j[1], P[i_j[0], i_j[1]]))
        else:
            for i_j in align_pairs:
                alignments.add('{0}-{1}'.format(i_j[0], i_j[1]))

        return alignments

    def get_khps_med(self, C1, C2, khp, vals_str):
        khp1 = C1.mean() #0.5*torch.median(C1.view(-1))
        khp2 = C2.mean() #0.5*torch.median(C2.view(-1))
        if "med_" in khp:
            val = float(vals_str[1])
            khp1 /= val
            khp2 /= val
        elif "_med" in khp:
            val = float(vals_str[0])
            khp1 *= val
            khp2 *= val
        return khp1, khp2

    def compute_optimal_transport(self, s1_word_embeddings, s2_word_embeddings):
        s1_word_embeddings = s1_word_embeddings.to(torch.float64)
        s2_word_embeddings = s2_word_embeddings.to(torch.float64)

        C = self.dist_func(s1_word_embeddings, s2_word_embeddings, self.distotion)
        s1_weights, s2_weights = self.weight_func(s1_word_embeddings, s2_word_embeddings)
        
        if self.s:
            self.s = int(C.shape[0]*C.shape[1]/20)*7

        if self.ktype == "lin":
            G1 = get_G(ktype="lin", x=s1_word_embeddings, y=s1_word_embeddings)
            G2 = get_G(ktype="lin", x=s2_word_embeddings, y=s2_word_embeddings)
        else:
            khp = self.khp
            median_heur = 0
            if isinstance(khp, str):
                median_heur = 1
                vals_str = None
                if "_" in khp:
                    vals_str = khp.split("_")
            else:
                khp1 = khp
                khp2 = khp
            
            C1 = 2*self.dist_func(s1_word_embeddings, s1_word_embeddings, 0)
            C2 = 2*self.dist_func(s2_word_embeddings, s2_word_embeddings, 0)                

            if median_heur:
                khp1, khp2 = self.get_khps_med(C1, C2, khp, vals_str)

            G1 = get_G(dist=C1, ktype=self.ktype, khp=khp1)
            G2 = get_G(dist=C2, ktype=self.ktype, khp=khp2)
            
            # print(torch.linalg.cond(G1), torch.linalg.cond(G2))

        P, S_i, S_j = get_gamma(C, G1, G2, s1_weights, s2_weights, self.max_itr, self.K, \
                                self.lda, self.lda3, all_gamma=self.all_gamma, conv_crit=1, s=self.s, ws=self.ws)
        
        # with open(self.logfile, 'a') as fp:
        #     fp.write(f"K chosen: {len(S_i)}\n")
            
        m, n = C.shape
        if self.all_gamma:
            P_list = []
            for i in range(len(P)):
                P_list.append(min_max_scaling(postprocess_gamma(P[i], S_i[:i+1], S_j[:i+1], m, n)))

            return P_list
        else:
        # Min-max normalization
            P = postprocess_gamma(P, S_i, S_j, m, n)
            P = min_max_scaling(P)
        return P

    def comvert_to_numpy(self, s1_weights, s2_weights, C):
        if torch.is_tensor(s1_weights):
            s1_weights = s1_weights.to('cpu').numpy()
            s2_weights = s2_weights.to('cpu').numpy()
        if torch.is_tensor(C):
            C = C.to('cpu').numpy()

        return s1_weights, s2_weights, C


    def bertscore_F1(self, vecX, vecY):
        vecX = F.normalize(vecX)
        vecY = F.normalize(vecY)
        matrix = torch.matmul(vecX, vecY.t())

        r = torch.sum(torch.amax(matrix, dim=1)) / matrix.shape[0]
        p = torch.sum(torch.amax(matrix, dim=0)) / matrix.shape[1]
        f = 2 * p * r / (p + r)

        return f.item()
