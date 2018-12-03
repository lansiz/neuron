# Latest paper is here: (https://arxiv.org/abs/1805.09001)(https://arxiv.org/abs/1805.09001)
# code for figures and experiments

## Figure 3: typical lambda function and their f.p.

`conn_09_typical_funcs_convergence.py` with different plasticity functions PF32, PF12, PF30, PF15.

## Figure 4: one-to-one mapping from stimulus to strength

`conn_08_s_x_relation_pf_30.py`

`conn_08_s_x_relation_pf_32.py`

## Figure 5: different stimulus

`conn_09_typical_funcs_convergence.py` with  plasticity functions PF12.

`conn_10_s_x_relation_sin_pf.py` to plot theta function.

## Figure 6: Theta function of discontinuous lambda

`conn_14_s_x_relation_discont_pf.py`

## Figure 7: lambda of choice

`conn_12_pf_of_year.py` constructs linear and threshhold-like thelta functions.

## Figure 9: neural network to f.p. on different lambda

`nn_02_different_e_to_fp.py` with different plasticity functions PF32, PF12, PF30, PF15.

## Figure 12: meshed-NN classifier for digit 6

`nn_meshed_dist_6_digit.py`

## Figure 13: meshed-NN classifier for all digits

`nn_meshed_dist_digit.py`

## Figure 15: Optimization of growable classifier

`nn_growable_6_digit_80_81.py`

## meshed-NN

training: `nn_meshed_train_batch.sh`

testing: `nn_meshed_test_batch.sh`

per-digit testing: `nn_meshed_test_number_batch.sh`

## growable-NN

training: `nn_growable_train_batch.sh`

testing: `nn_growable_test_batch.sh`
