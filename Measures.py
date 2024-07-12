import numpy as np
import AuxiliaryFunctions as af
from math import log2


def calc_meas(p_sca, p_s, p_s_pred, p_c, p_a):
    p_ct = np.zeros(pow(2,2))
    p_cj_scj = np.zeros((2, pow(2,5)))
    p_sncj = np.zeros((2, pow(2, 4)))

    p_cn_c = np.zeros((2, pow(2,3)))
    p_sn_a = np.zeros(pow(2,5))
    p_sn_a_noncond = np.zeros(pow(2, 5))

    p_full = np.zeros(pow(2,14))
    p_mutful = np.zeros(pow(2,14))

    p_full_predict = np.zeros(pow(2,10))
    p_full_predict2 = np.zeros(pow(2,7))

    p_s_c_a_pred = np.zeros(pow(2,5))
    p_s_c_c_pred = np.zeros(pow(2,5))
    p_sa = np.zeros(pow(2,5))
    p_sc = np.zeros(pow(2,5))
    p_ca = np.zeros(pow(2,4))
    p_s_s = np.zeros(pow(2,6))
    p_s_s_noncond = np.zeros(pow(2,6))
    p_s_t = np.zeros(pow(2,3))

    p_at = np.zeros(pow(2,2))
    p_an_c = np.zeros((2, pow(2,3)))
    p_an_s = np.zeros((2, pow(2,4)))

    p_a_n = np.zeros(pow(2,2))
    p_c_n = np.zeros(pow(2,2))
    p_s_n = np.zeros(pow(2,3))
    p_s_n_pred = np.zeros(pow(2, 3))
    p_s_pred2 = np.zeros(pow(2,7))
    p_s_pred2_sgoal = np.zeros(pow(2,6))
    p_s_sa = np.zeros(pow(2, 8))


    p_goal = [0.0,0.0]
    p_goal_pred = [0.0,0.0]


    for i in range(pow(2,14)):
        p_full[i] = p_sca[i//pow(2,7)]*p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*\
                    p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,2))] * \
                    p_s[(i//pow(2,4))%pow(2,5) + pow(2,5) *(i//pow(2,11))]*\
                    p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]
    #distr on var only in time point t
    for i in range(pow(2,10)):
        p_full_predict[i] = p_sca[i//pow(2,3)] * p_s_pred[i%pow(2,7)]
    for i in range(pow(2,10)):
        p_s_c_a_pred[i%pow(2,5)] = p_s_c_a_pred[i%pow(2,5)] + p_full_predict[i]
        p_s_c_c_pred[i%pow(2,3) + pow(2,3)*((i//pow(2,5))%pow(2,2))] = p_s_c_c_pred[i%pow(2,3) + pow(2,3)*((i//pow(2,5))%pow(2,2))] + p_full_predict[i]
        p_s_pred2[i%pow(2,7)] = p_s_pred2[i%pow(2,7)] + p_full_predict[i]
        p_s_pred2_sgoal[(i//2)%pow(2,6)] = p_s_pred2_sgoal[(i//2)%pow(2,6)] + p_full_predict[i]
        p_goal_pred[i%2] = p_goal_pred[i%2] + p_full_predict[i]
        p_full_predict2[i%pow(2,7)] = p_full_predict2[i%pow(2,7)]+ p_full_predict[i]
        p_s_n_pred[i%pow(2,3)] = p_s_n_pred[i%pow(2,3)] + p_full_predict[i]

    p_s_c_a_prednoncond = np.copy(p_s_c_a_pred)
    p_s_c_c_prednoncond = np.copy(p_s_c_c_pred)
    p_marg_predict = np.copy(p_s_pred2)

    for i in range(pow(2,7)):
        p_at[i%pow(2,2)] = p_at[i%pow(2,2)] + p_sca[i]
        p_ct[(i//pow(2,2))%pow(2,2)] = p_ct[(i//pow(2,2))%pow(2,2)] + p_sca[i]
        p_sa[i%pow(2,2) + pow(2,2) * ((i//pow(2,4))%pow(2,3))] =  p_sa[i%pow(2,2) + pow(2,2) * ((i//pow(2,4))%pow(2,3))] + p_sca[i]
        p_ca[i%pow(2,4)] = p_ca[i%pow(2,4)] + p_sca[i]
        p_sc[i // pow(2, 2)] = p_sc[i // pow(2, 2)] + p_sca[i]
    #distr on both var
    for i in range(pow(2,5)):
        p_s_c_a_pred[i] = p_s_c_a_pred[i]/ p_at[i//pow(2,3)]
        p_s_c_c_pred[i] = p_s_c_c_pred[i]/ p_ct[i//pow(2,3)]
    for i in range(pow(2,7)):
        p_s_pred2[i] = p_s_pred2[i] /p_ca[i//pow(2,3)]

    for i in range(pow(2,6)):
        p_s_pred2_sgoal[i] = p_s_pred2_sgoal[i] /p_ca[i//pow(2,2)]

    for i in range(pow(2,14)):
        p_cj_scj[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%2)] =p_cj_scj[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%2)] + p_full[i]
        p_cj_scj[1][(i // pow(2, 2)) % 2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)] = p_cj_scj[1][(i // pow(2, 2)) % 2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]+ p_full[i]
        p_sncj[0][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,10))%2)] = p_sncj[0][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,10))%2)]+ p_full[i]
        p_sncj[1][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,9))%2)] = p_sncj[1][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,9))%2)] + p_full[i]

        p_sn_a[(i//pow(2,4))%pow(2,5)] = p_sn_a[(i//pow(2,4))%pow(2,5)] +p_full[i]
        p_sn_a_noncond[(i // pow(2, 4)) % pow(2, 5)] = p_sn_a_noncond[(i // pow(2, 4)) % pow(2, 5)] + p_full[i]
        p_s_s[(i//pow(2,4))%pow(2,3) + pow(2,3)*(i//pow(2,11))] = p_s_s[(i//pow(2,4))%pow(2,3) + pow(2,3)*(i//pow(2,11))] + p_full[i]

        p_s_s_noncond[(i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11))] = p_s_s_noncond[(i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * ( i // pow(2, 11))] + p_full[i]
        p_s_t[i//pow(2,11)] =  p_s_t[i//pow(2,11)] + p_full[i]

        p_s_sa[(i//pow(2,4))%pow(2,5)+ pow(2,5)*(i//pow(2,11))] = p_s_sa[(i//pow(2,4))%pow(2,5)+ pow(2,5)*(i//pow(2,11))] + p_full[i]

        p_a_n[i%pow(2,2)] = p_a_n[i%pow(2,2)] +p_full[i]
        p_s_n[(i//pow(2,4))%pow(2,3)] = p_s_n[(i//pow(2,4))%pow(2,3)] + p_full[i]
        p_c_n[(i//pow(2,2))%pow(2,2)] = p_c_n[(i//pow(2,2))%pow(2,2)] + p_full[i]

        p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9))%pow(2,2))] = p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9))%pow(2,2))] +p_full[i]
        p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 2))] = p_cn_c[1][(i//pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 2))] + p_full[i]

        p_an_c[0][(i//2)%pow(2,3)] = p_an_c[0][(i//2)%pow(2,3)] +p_full[i]
        p_an_c[1][ i%2 + 2*((i // 4) % pow(2, 2))] = p_an_c[1][ i%2 + 2*((i // 4) % pow(2, 2))]+ p_full[i]

        p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))] = p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))] + p_full[i]
        p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] = p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] + p_full[i]
        p_goal[(i // pow(2, 4)) % 2] = p_goal[(i // pow(2, 4) )% 2] + p_full[i]
    for i in range(pow(2,14)):
        p_mutful[i] = p_sca[i // pow(2, 7)] * p_c_n[(i // pow(2, 2)) % pow(2, 2)] * p_s_n[(i // pow(2, 4)) % pow(2, 3)] * p_a_n[i % pow(2, 2)]
    #conditioning
    for i in range(pow(2,6)):
        if p_s_t[i//pow(2,3)] > 0:
            p_s_s[i] = p_s_s[i] / p_s_t[i//pow(2,3)]
        else:
            p_s_s[i] = 1/8

    for i in range(pow(2,5)):
        for j in range(2):
            if p_sncj[j][i//2] > 0:
                p_cj_scj[j][i] = p_cj_scj[j][i] / p_sncj[j][i//2]
            else:
                p_cj_scj[j][i] = 0.5
        if p_at[i//pow(2,3)] > 0:
            p_sn_a[i] = p_sn_a[i] / p_at[i//pow(2,3)]
        else:
            p_sn_a[i] = 0.5

    for i in range(pow(2,4)):
        for j in range(2):
            if p_s_n[i//2] > 0:
                p_an_s[j][i] = p_an_s[j][i] / p_s_n[i//2]
 #   ###print("t", np.sum(p_an_s), np.sum(p_s_n))
    for i in range(pow(2,3)):
        for j in range(2):
            if p_c_n[i//2]>0:
                p_an_c[j][i] = p_an_c[j][i] / p_c_n[i//2]
            else:
                p_an_c[j][i] = 0.5
            if p_ct[i//2]>0:
                p_cn_c[j][i] =  p_cn_c[j][i] / p_ct[i//2]
            else:
                p_cn_c[j][i] = 0.5
   # ###print("t", np.sum(p_cn_c))
    int_int = 0.0
    memory = 0.0
    morph = 0.0
    morphfull = 0.0
    react = 0.0
    sensi = 0.0
    muti = 0.0
    comm = 0.0
    pred1 = 0.0
    pred2 = 0.0
    predfull = 0.0
    predgoal =  0.0
    action = 0.0

    for i in range(pow(2,7)):
        if p_s_pred2[ i]*p_s_c_a_pred[i%pow(2,5)] > 0:
            pred1 = pred1+ p_marg_predict[i] * (log2(p_s_pred2[ i]/p_s_c_a_pred[i%pow(2,5)]))
        if p_s_pred[ i%pow(2,7)]* p_s_c_c_pred[i%pow(2,3) + pow(2,3)*((i//pow(2,5))%pow(2,2))] > 0:
            pred2 = pred2 + p_marg_predict[i] * (log2(p_s_pred2[ i] /p_s_c_c_pred[i%pow(2,3) + pow(2,3)*(i//pow(2,5))]))
        if p_s_pred2[i % pow(2, 7)] * p_s_n[i % pow(2, 3)] > 0:
            predfull = predfull + p_marg_predict[i] * (log2(p_s_pred2[i]/p_s_n_pred[i % pow(2, 3)]))
        if p_s_pred2[i % pow(2, 7)] * p_s_n[i % pow(2, 3)] > 0:
            predgoal = predgoal + p_marg_predict[i] * (log2(p_s_pred2[i]/(p_s_pred2_sgoal[(i//2)%pow(2,6)]* p_goal_pred[i%2])))


    for i in range(pow(2,14)):
        if (p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3) )+ pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_cj_scj[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % 2)]*  p_cj_scj[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]) >0 :
                int_int = int_int + p_full[i]* (log2(p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,2))]) -
                                                log2(p_cj_scj[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % 2)]*  p_cj_scj[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]))
                memory = memory + p_full[i]*(log2(p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,2))])- log2(p_c_n[(i//pow(2,2))%pow(2,2)]))
        if (p_c[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 9)) % pow(2, 2))] * p_c[1][ (i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ( (i // pow(2, 9)) % pow(2, 2))]*p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9))%pow(2,2))]*  p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 2))]) > 0:
            sensi = sensi + p_full[i]*  (log2(p_c[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,9))%pow(2,2))]*p_c[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,2))]) -log2(p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,9))%pow(2,2))]*  p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 2))]))

        if  p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))]*p_sn_a[(i//pow(2,4))%pow(2,5)] > 0:
            morph = morph + p_full[i] * (log2(p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))])-log2(p_sn_a[(i//pow(2,4))%pow(2,5)] ) )
            morphfull = morphfull + p_full[i] *(log2(p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))])-log2(p_s_n[(i//pow(2,4))%pow(2,3)] ) )
        if p_full[i]*p_mutful[i] > 0:
            muti = muti + p_full[i]*(log2(p_full[i])- log2(p_mutful[i]))
        if p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]* p_an_c[0][(i//2)%pow(2,3)] *p_an_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))] >0:
            react = react + p_full[i]*(log2(p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))])-
                                       log2(p_an_c[0][(i//2)%pow(2,3)] *p_an_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))] ) )
        if p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))]   * p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))]*p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]  > 0:
            comm = comm + p_full[i] * (log2(p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))])-
                                       log2(p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))]   * p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] ))
        if p_s_s[((i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11)))] * p_s[(i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11))] > 0:
            action = action + p_full[i] * (log2(p_s[(i // pow(2, 4)) % pow(2, 5) + pow(2, 5) * (i // pow(2, 11))]) - log2(p_s_s[((i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11)))]))

    #iterative scaling
    p_it = np.zeros(pow(2, 7)) + (1 / pow(2, 7))
    diff = 1
    syn = 0.0
    synmorph = 0.0
    #synergistic prediciton measure
    synergistic = False
    if synergistic == True:
        while diff > 0.000001:
            oldp = np.copy(p_it)
            pit_ct_at = np.zeros(pow(2, 4))
            for j in range(pow(2, 7)):
                pit_ct_at[(j // pow(2, 3))] = pit_ct_at[(j // pow(2, 3))] + p_it[j]
            for j in range(pow(2, 7)):
                p_it[j] = p_it[j] * (p_ca[(j // pow(2, 3))] / pit_ct_at[(j // pow(2, 3))])
            pit_s_a = np.zeros(pow(2, 5))
            print(1, np.sum(p_it))
            for j in range(pow(2, 7)):
                pit_s_a[j % pow(2, 5)] = pit_s_a[j % pow(2, 5)] + p_it[j]
            for j in range(pow(2, 7)):
                p_it[j] = p_it[j] * (p_s_c_a_prednoncond[j % pow(2, 5)] / pit_s_a[j % pow(2, 5)])
            print(2, np.sum(p_it))
            pit_s_c = np.zeros(pow(2, 5))
            for j in range(pow(2, 7)):
                pit_s_c[j % pow(2, 3) + pow(2, 3) * ((j // pow(2, 5)) % pow(2, 2))] = pit_s_c[j % pow(2, 3) + pow(2, 3) * (
                            (j // pow(2, 5)) % pow(2, 2))] + p_it[j]
            for j in range(pow(2, 7)):
                    p_it[j] = p_it[j] * (p_s_c_c_prednoncond[j % pow(2, 3) + pow(2, 3) * ((j // pow(2, 5)) % pow(2, 2))] / pit_s_c[
                        j % pow(2, 3) + pow(2, 3) * ((j // pow(2, 5)) % pow(2, 2))])
            diff = af.kl(oldp, p_it)
            print(3, np.sum(p_it))
        syn = af.kl(p_full_predict2, p_it)
    #synergistic Morphological
    p_it = np.zeros(pow(2, 8)) + (1 / pow(2, 8))
    diff = 1
    synergisticmorph = True
    if synergisticmorph == True:
        while diff > 0.000001:
            oldp = np.copy(p_it)
            pit_st_st = np.zeros(pow(2, 6))
            for j in range(pow(2, 8)):
                pit_st_st[(j % pow(2, 3))+ (pow(2,3))*(j//pow(2,5))] = pit_st_st[(j % pow(2, 3))+ (pow(2,3))*(j//pow(2,5))] + p_it[j]
            for j in range(pow(2, 8)):
                p_it[j] = p_it[j] * (p_s_s_noncond[(j % pow(2, 3))+ (pow(2,3))*(j//pow(2,5))] / pit_st_st[(j % pow(2, 3))+ (pow(2,3))*(j//pow(2,5))])

            pit_s_a = np.zeros(pow(2, 5))
            for j in range(pow(2, 8)):
                pit_s_a[j % pow(2, 5)] = pit_s_a[j % pow(2, 5)] + p_it[j]

            for j in range(pow(2, 8)):
                p_it[j] = p_it[j] * (p_sn_a_noncond[j % pow(2, 5)] / pit_s_a[j % pow(2, 5)])

            pit_sa = np.zeros(pow(2, 5))

            for j in range(pow(2, 8)):
                pit_sa[j // pow(2, 3)] = pit_sa[j // pow(2, 3)] + p_it[j]
            for j in range(pow(2, 8)):
                    p_it[j] = p_it[j] * (p_sa[j //pow(2,3)] / pit_sa[j//pow(2,3)])
            diff = af.kl(oldp, p_it)
        synmorph = af.kl(p_s_sa, p_it)
        print("syn", syn, pred1, pred2, predfull, "morph", synmorph, morph, action, morphfull, "int", int_int, memory)
    return int_int, morph, react, sensi, muti, comm, p_goal[1], pred1, pred2, syn,  predfull, predgoal, action, synmorph, morphfull, memory

def calc_meas_noint(p_sca, p_s, p_s_pred, p_c_i, p_a):

    p_ct = np.zeros((2,2))
    p_ct2 = np.zeros(pow(2, 2))
    p_cj_scj = np.zeros((2, pow(2,5)))
    p_sncj = np.zeros((2, pow(2, 4)))

    p_cn_c = np.zeros((2, pow(2,2)))
    p_sn_a = np.zeros(pow(2,5))

    p_full = np.zeros(pow(2,14))
    p_mutful = np.zeros(pow(2,14))

    p_at = np.zeros(pow(2,2))
    p_an_c = np.zeros((2, pow(2,3)))
    p_an_s = np.zeros((2, pow(2,4)))

    p_a_n = np.zeros(pow(2,2))
    p_c_n = np.zeros(pow(2,2))
    p_s_n = np.zeros(pow(2,3))
    p_sa = np.zeros(pow(2, 5))
    p_s_s = np.zeros(pow(2,6))
    p_s_t = np.zeros(pow(2,3))


    p_ca = np.zeros(pow(2, 5))


    p_s_pred2_sgoal = np.zeros(pow(2, 6))

    p_full_predict = np.zeros(pow(2,10))
    p_full_predict2 = np.zeros(pow(2,7))
    p_s_c_a_pred = np.zeros(pow(2,5))
    p_s_c_c_pred = np.zeros(pow(2,5))


    p_goal_pred = [0.0, 0.0]
    p_goal = [0.0,0.0]


    for i in range(pow(2,14)):
        p_full[i] = p_sca[i//pow(2,7)]*p_c_i[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%pow(2,1))]*\
                    p_c_i[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,1))] * \
                    p_s[(i//pow(2,4))%pow(2,5) + pow(2,5) *(i//pow(2,11))]* \
                    p_a[0][(i // 2) % pow(2, 6)] * p_a[1][i % 2 + 2 * ((i // pow(2, 2)) % pow(2, 5))]
    #distr on var only in time point t
    for i in range(pow(2,7)):
        p_at[i%pow(2,2)] = p_at[i%pow(2,2)] + p_sca[i]
        p_ct2[(i // pow(2, 2)) % pow(2, 2)] = p_ct2[(i // pow(2, 2)) % pow(2, 2)] + p_sca[i]
        p_ct[0][(i//pow(2,3))%pow(2,1)] = p_ct[0][(i//pow(2,3))%pow(2,1)] + p_sca[i]
        p_ct[1][(i//pow(2,2))%pow(2,1)] = p_ct[1][(i//pow(2,2))%pow(2,1)] + p_sca[i]
        p_sa[i % pow(2, 2) + pow(2, 2) * ((i // pow(2, 4)) % pow(2, 3))] = p_sa[i % pow(2, 2) + pow(2, 2) * ( (i // pow(2, 4)) % pow(2, 3))] + p_sca[i]
        p_ca[i % pow(2, 4)] = p_ca[i % pow(2, 4)] + p_sca[i]

    for i in range(pow(2, 10)):
        p_full_predict[i] = p_sca[i // pow(2, 3)] * p_s_pred[i % pow(2, 7)]
    for i in range(pow(2, 10)):
        p_s_c_a_pred[i % pow(2, 5)] = p_s_c_a_pred[i % pow(2, 5)] + p_full_predict[i]
        p_s_c_c_pred[i % pow(2, 3) + pow(2, 3) * ((i // pow(2, 5)) % pow(2, 2))] = p_s_c_c_pred[i % pow(2, 3) + pow(2, 3) * ((i // pow(2,5)) % pow(2, 2))] + p_full_predict[i]
        p_s_pred2_sgoal[(i // 2) % pow(2, 6)] = p_s_pred2_sgoal[(i // 2) % pow(2, 6)] + p_full_predict[i]
        p_goal_pred[i % 2] = p_goal_pred[i % 2] + p_full_predict[i]
        p_full_predict2[i % pow(2, 7)] = p_full_predict2[i % pow(2, 7)] + p_full_predict[i]

    for i in range(pow(2,4)):
        p_s_c_a_pred[i] = p_s_c_a_pred[i]/ p_at[i//pow(2,3)]
        p_s_c_c_pred[i] = p_s_c_c_pred[i]/ p_ct2[i//pow(2,3)]
    for i in range(pow(2,6)):
        p_s_pred2_sgoal[i] = p_s_pred2_sgoal[i] /p_ca[i//pow(2,2)]
    #distr on both var
    for i in range(pow(2,14)):
        p_cj_scj[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%2)] =p_cj_scj[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%2)] + p_full[i]
        p_cj_scj[1][(i // pow(2, 2)) % 2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)] = p_cj_scj[1][(i // pow(2, 2)) % 2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]+ p_full[i]
        p_sncj[0][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,10))%2)] = p_sncj[0][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,10))%2)]+ p_full[i]
        p_sncj[1][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,9))%2)] = p_sncj[1][(i//pow(2,4))%pow(2,3) + pow(2,3)*((i//pow(2,9))%2)] + p_full[i]

        p_sn_a[(i // pow(2, 4)) % pow(2, 5)] = p_sn_a[(i // pow(2, 4)) % pow(2, 5)] + p_full[i]
        p_s_s[(i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11))] = p_s_s[(i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11))] + p_full[i]
        p_s_t[i // pow(2, 11)] = p_s_t[i // pow(2, 11)] + p_full[i]

        p_a_n[i%pow(2,2)] = p_a_n[i%pow(2,2)] +p_full[i]
        p_s_n[(i//pow(2,4))%pow(2,3)] = p_s_n[(i//pow(2,4))%pow(2,3)] + p_full[i]
        p_c_n[(i//pow(2,2))%pow(2,2)] = p_c_n[(i//pow(2,2))%pow(2,2)] + p_full[i]

        p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,10))%pow(2,1))] = p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,10))%pow(2,1))] +p_full[i]
        p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 1))] = p_cn_c[1][(i//pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 1))] + p_full[i]

        p_an_c[0][(i//2)%pow(2,3)] = p_an_c[0][(i//2)%pow(2,3)] +p_full[i]
        p_an_c[1][ i%2 + 2*((i // 4) % pow(2, 2))] = p_an_c[1][ i%2 + 2*((i // 4) % pow(2, 2))]+ p_full[i]

        p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))] = p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))] + p_full[i]
        p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] = p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] + p_full[i]
        p_goal[(i // pow(2, 4)) % 2] = p_goal[(i // pow(2, 4) )% 2] + p_full[i]
    for i in range(pow(2,14)):
        p_mutful[i] = p_sca[i // pow(2, 7)] * p_c_n[(i // pow(2, 2)) % pow(2, 2)] * p_s_n[(i // pow(2, 4)) % pow(2, 3)] * p_a_n[i % pow(2, 2)]
    #conditioning

    for i in range(pow(2,6)):
        if p_s_t[i//pow(2,3)] > 0:
            p_s_s[i] = p_s_s[i] / p_s_t[i//pow(2,3)]
        else:
            p_s_s[i] = 1/8
    for i in range(pow(2,5)):
        for j in range(2):
            if p_sncj[j][i//2] > 0:
                p_cj_scj[j][i] = p_cj_scj[j][i] / p_sncj[j][i//2]
            else:
                p_cj_scj[j][i] = 0.5
        if p_at[i//pow(2,3)] > 0:
            p_sn_a[i] = p_sn_a[i] / p_at[i//pow(2,3)]
        else:
            p_sn_a[i] = 0.5

    for i in range(pow(2,4)):
        for j in range(2):
            if p_s_n[i//2] > 0:
                p_an_s[j][i] = p_an_s[j][i] / p_s_n[i//2]
 #   ###print("t", np.sum(p_an_s), np.sum(p_s_n))
    for i in range(pow(2,3)):
        for j in range(2):
            if p_c_n[i//2]>0:
                p_an_c[j][i] = p_an_c[j][i] / p_c_n[i//2]
            else:
                p_an_c[j][i] = 0.5
    for i in range(pow(2, 2)):
        for j in range(2):
            if p_ct[j][i//2]>0:
                p_cn_c[j][i] =  p_cn_c[j][i] / p_ct[j][i//2]
            else:
                p_cn_c[j][i] = 0.5
   # ###print("t", np.sum(p_cn_c))
    int_int = 0.0
    morph = 0.0
    react = 0.0
    sensi = 0.0
    muti = 0.0
    comm = 0.0
    action = 0.0

    pred1 = 0.0
    pred2 = 0.0
    predfull = 0.0
    predgoal = 0.0
    for i in range(pow(2, 10)):
        if p_s_pred[i % pow(2, 7)] * p_s_c_a_pred[i % pow(2, 5)] > 0:
            pred1 = pred1 + p_full_predict[i] * (log2(p_s_pred[i % pow(2, 7)]) - log2(p_s_c_a_pred[i % pow(2, 5)]))
        if p_s_pred[i % pow(2, 7)] * p_s_c_c_pred[i % pow(2, 3) + pow(2, 3) * ((i // pow(2, 5)) % pow(2, 2))] > 0:
            pred2 = pred2 + p_full_predict[i] * (log2(p_s_pred[i % pow(2, 7)]) - log2(
                p_s_c_c_pred[i % pow(2, 3) + pow(2, 3) * ((i // pow(2, 5)) % pow(2, 2))]))
        if p_s_pred[i % pow(2, 7)] * p_s_n[i % pow(2, 3)] > 0:
            predfull = predfull + p_full_predict[i] * (log2(p_s_pred[i % pow(2, 7)]) - log2(p_s_n[i % pow(2, 3)]))
        if p_s_pred[i % pow(2, 7)] * p_s_n[i % pow(2, 3)] > 0:
            predgoal = predgoal +  p_full_predict[i] * (log2(p_s_pred[i%pow(2,7)]/(p_s_pred2_sgoal[(i//2)%pow(2,6)]* p_goal_pred[i%2])))

    for i in range(pow(2,14)):
         if (p_c_i[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%pow(2,1))]*p_c_i[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3) )+ pow(2,4)*((i//pow(2,9))%pow(2,1))]*p_cj_scj[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % 2)]*  p_cj_scj[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]) >0 :
                int_int = int_int + p_full[i]* (log2(p_c_i[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%pow(2,1))]*p_c_i[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,1))]) -
                                                log2(p_cj_scj[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % 2)]*  p_cj_scj[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ((i // pow(2, 9)) % 2)]))
         if (p_c_i[0][(i // pow(2, 3)) % pow(2, 4) + pow(2, 4) * ((i // pow(2, 10)) % pow(2, 1))] * p_c_i[1][ (i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 4)) % pow(2, 3)) + pow(2, 4) * ( (i // pow(2, 9)) % pow(2, 1))]*p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,10))%pow(2,1))]*  p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 1))]) > 0:
            sensi = sensi + p_full[i]*  (log2(p_c_i[0][(i//pow(2,3))%pow(2,4) + pow(2,4)*((i//pow(2,10))%pow(2,1))]*p_c_i[1][(i//pow(2,2))%2 + 2*((i//pow(2,4))%pow(2,3)) + pow(2,4)*((i//pow(2,9))%pow(2,1))]) -log2(p_cn_c[0][(i//pow(2,3))%2 + 2*((i//pow(2,10))%pow(2,1))]*  p_cn_c[1][(i // pow(2, 2)) % 2 + 2 * ((i // pow(2, 9)) % pow(2, 1))]))
         if  p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))] > 0:
            morph = morph + p_full[i] * (log2(p_s[(i//pow(2,4))%pow(2,5) + pow(2,5)*(i//pow(2,11))])-log2(p_sn_a[(i//pow(2,4))%pow(2,5)] ) )
         if p_full[i]*p_mutful[i] > 0:
            muti = muti + p_full[i]*(log2(p_full[i])- log2(p_mutful[i]))
         if p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]* p_an_c[0][(i//2)%pow(2,3)] *p_an_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))] >0:
            react = react + p_full[i]*(log2(p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))])-
                                       log2(p_an_c[0][(i//2)%pow(2,3)] *p_an_c[1][i%2 + 2*((i//pow(2,2))%pow(2,2))] ) )
         if p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))]   * p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))]*p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))]  > 0:
            comm = comm + p_full[i] * (log2(p_a[0][(i//2)%pow(2,6)]*p_a[1][i%2 + 2*((i//pow(2,2))%pow(2,5))])-
                                       log2(p_an_s[0][(i//2)%2 + 2* ((i//pow(2,4))%pow(2,3))]   * p_an_s[1][i%2 + 2*((i//pow(2,4)) %pow(2,3))] ))
         if p_s_s[((i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11)))] * p_s[(i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11))] > 0:
            action = action + p_full[i] * (log2(p_s[(i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11))]) - log2(p_s_s[((i // pow(2, 4)) % pow(2, 3) + pow(2, 3) * (i // pow(2, 11)))]))


    #iterative scaling
    p_it = np.zeros(pow(2, 7)) + (1 / pow(2, 7))
    diff = 1
    # synergistic prediction measure
    synergistic = False
    syn = 0.0
    if synergistic == True:
        while diff > 0.000001:
            oldp = np.copy(p_it)
            pit_ct_at = np.zeros(pow(2, 4))
            for j in range(pow(2, 7)):
                pit_ct_at[(j // pow(2, 3))] = pit_ct_at[(j // pow(2, 3))] + p_it[j]
            for j in range(pow(2, 7)):
                p_it[j] = p_it[j] * (p_ca[(j // pow(2, 3))] / pit_ct_at[(j // pow(2, 3))])
            pit_s_a = np.zeros(pow(2, 5))
            pit_a = np.zeros(pow(2, 2))
            for j in range(pow(2, 7)):
                pit_s_a[j % pow(2, 5)] = pit_s_a[j % pow(2, 5)] + p_it[j]
                pit_a[(j // pow(2, 3)) % pow(2, 2)] = pit_a[(j // pow(2, 3)) % pow(2, 2)] + p_it[j]
            for j in range(pow(2, 5)):
                pit_s_a[j] = pit_s_a[j] / pit_a[j // pow(2, 3)]

            for j in range(pow(2, 7)):
                p_it[j] = p_it[j] * (p_s_c_a_pred[j % pow(2, 5)] / pit_s_a[j % pow(2, 5)])

            pit_s_c = np.zeros(pow(2, 5))
            pit_c = np.zeros(pow(2, 2))
            for j in range(pow(2, 7)):
                pit_s_c[j % pow(2, 3) + pow(2, 3) * ((j // pow(2, 5)) % pow(2, 2))] = pit_s_c[j % pow(2, 3) + pow(2, 3) * (
                            (j // pow(2, 5)) % pow(2, 2))] + p_it[j]
                pit_c[((j // pow(2, 5)) % pow(2, 2))] = pit_c[((j // pow(2, 5)) % pow(2, 2))] + p_it[j]
            for j in range(pow(2, 5)):
                pit_s_c[j] = pit_s_c[j] / pit_c[j // pow(2, 3)]
            for j in range(pow(2, 7)):
                if pit_s_c[j % pow(2, 3) + pow(2, 3) * ((j // pow(2, 5)) % pow(2, 2))] > 0:
                    p_it[j] = p_it[j] * (p_s_c_c_pred[j % pow(2, 3) + pow(2, 3) * ((j // pow(2, 5)) % pow(2, 2))] / pit_s_c[
                        j % pow(2, 3) + pow(2, 3) * ((j // pow(2, 5)) % pow(2, 2))])
                else:
                    p_it[j] = p_it[j] * (1 / 8)
            diff = af.kl(oldp, p_it)
        syn = af.kl(p_full_predict2, p_it)
    return int_int, morph, react, sensi, muti, comm, p_goal[1], pred1, pred2, syn, predfull, predgoal, action

def calc_meas_morph(p_sa, p_s):

    p_sn_a = np.zeros(pow(2,5))
    p_sn_s = np.zeros(pow(2,6))

    p_full = np.zeros(pow(2,8))

    p_at = np.zeros(pow(2,2))
    p_st = np.zeros(pow(2,3))

    for i in range(pow(2,8)):
        p_full[i] = p_sa[i//pow(2,3)]*p_s[i]

    print("full", np.sum(p_full))
    for i in range(pow(2,5)):
        p_at[i%pow(2,2)] = p_at[i%pow(2,2)] + p_sa[i]
        p_st[i//pow(2,2)] = p_st[i//pow(2,2)] + p_sa[i]

    #distr on both var
    for i in range(pow(2,8)):
        p_sn_a[i  % pow(2, 5)] = p_sn_a[i  % pow(2, 5)] + p_full[i]
        p_sn_s[i  % pow(2, 3) + pow(2,3)*(i//pow(2,5))] = p_sn_s[i  % pow(2, 3) + pow(2,3)*(i//pow(2,5))] + p_full[i]

    p_sn_a_joint = np.copy(p_sn_a)
    p_sn_s_joint = np.copy(p_sn_s)

    for i in range(pow(2,5)):
        if p_at[i//pow(2,3)] > 0:
            p_sn_a[i] = p_sn_a[i] / p_at[i//pow(2,3)]
        else:
            p_sn_a[i] = 1/8

    for i in range(pow(2,6)):
        if p_st[i//pow(2,3)] > 0:
            p_sn_s[i] = p_sn_s[i] / p_st[i//pow(2,3)]
        else:
            p_sn_s[i] = 1/8

    morph = 0.0
    actione = 0.0
    for i in range(pow(2,8)):
         if  p_s[i] > 0:
            morph = morph + p_full[i] * (log2(p_s[i])-log2(p_sn_a[(i)%pow(2,5)] ) )
            actione = actione + p_full[i] *(log2(p_s[i]) - log2(p_sn_s[i  % pow(2, 3) + pow(2,3)*(i//pow(2,5))]))
    print("Morph", morph, actione, np.sum(p_full), np.sum(p_s), np.sum(p_sn_a), np.sum(p_sn_s))

    syn = 0.0
    pit = np.ones(pow(2,8))/pow(2,8)
    for j in range(100):
        pit_sn_a = np.zeros(pow(2,5))
        for i in range(pow(2, 8)):
            pit_sn_a[i % pow(2, 5)] = pit_sn_a[i % pow(2, 5)] + pit[i]
        for i in range(pow(2, 8)):
            pit[i] = pit[i] * (p_sn_a_joint[i % pow(2, 5)] / pit_sn_a[i % pow(2, 5)])

        pit_sn_s = np.zeros(pow(2,6))
        for i in range(pow(2, 8)):
            pit_sn_s[i % pow(2, 3) + pow(2, 3) * (i // pow(2, 5))] = pit_sn_s[i % pow(2, 3) + pow(2, 3) * (i // pow(2, 5))] + pit[i]
        for i in range(pow(2, 8)):
            pit[i] = pit[i] * (p_sn_s_joint[i % pow(2, 3) + pow(2, 3) * (i // pow(2, 5))] / pit_sn_s[i % pow(2, 3) + pow(2, 3) * (i // pow(2, 5))])

        pit_s_a = np.zeros(pow(2,5))
        for i in range(pow(2,8)):
            pit_s_a[i//pow(2,3)] = pit_s_a[i//pow(2,3)] + pit[i]
        for i in range(pow(2, 8)):
            pit[i] = pit[i] * (p_sa[i // pow(2, 3)] / pit_s_a[i // pow(2, 3)])

    for i in range(pow(2,8)):
        if p_s[i]>0:
            syn = syn + p_full[i] * (log2(p_full[i])- log2(pit[i]))

    return morph, actione, syn
