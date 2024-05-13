import torch
import torch.nn.functional as F
import numpy as np
import ot
from util import *
from mparts import *


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
        
        P = ot.smooth.smooth_ot_dual(s1_weights.cpu().numpy(), s2_weights.cpu().numpy(), C.cpu().numpy(), self.lda)
            
        P = min_max_scaling(torch.from_numpy(P).to(C.device))
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
