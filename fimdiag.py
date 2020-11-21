import numpy as np

# Functions to compute FIM diagonal based on samples from RBMs

def rbm_fim_diag(sample, nvis):
    # only works for small models and/or few samples
    sample = np.asarray(sample).astype(bool)
    nsamples = sample.shape[0]
    vis = sample[:, :nvis]
    hid = sample[:, nvis:]
    prod = vis[:, :, np.newaxis] * hid[:, np.newaxis, :]
    s = np.hstack([vis, hid, prod.reshape((nsamples, -1))])
    return np.var(s, axis=0, ddof=1)


def rbm_fi_biases(sample, nvis):
    # computes FIM diagnoal element for the visible and hidden biases, column-wise
    sample = np.asarray(sample).astype(bool)
    nsamples = sample.shape[0]
    vis = sample[:, :nvis]
    hid = sample[:, nvis:]
    prod = (vis[:, :, np.newaxis] * hid[:, np.newaxis, :]).reshape((nsamples, -1))
    
    # compute variance for vis biases
    square_elem = np.square(vis)
    sum_of_square = np.sum(square_elem, axis =0)
    sum_col = np.sum(vis, axis=0)
    square_of_sums = np.square(sum_col)
    mean_of_sq_of_sums = square_of_sums / nsamples
    
    res_vis = np.asarray((sum_of_square - mean_of_sq_of_sums)/(nsamples-1))
        
    # compute variance for hid biases
    square_elem = np.square(hid)
    sum_of_square = np.sum(square_elem, axis =0)
    sum_col = np.sum(hid, axis=0)
    square_of_sums = np.square(sum_col)
    mean_of_sq_of_sums = square_of_sums / nsamples
    
    res_hid = np.asarray((sum_of_square - mean_of_sq_of_sums)/(nsamples-1))   

    res = np.hstack([res_vis, res_hid])
    return res

def FI_weights_var_heur_estimates(samples, nv, nh, weights, mask=None):
    # compute variance and heuristic estimates of FI for the weights, not biases
    def f(x):
        return 1/(1+np.exp(-x))

    def finv(p):
        return -np.log(1/p-1)

    sample2, nvis2 = samples,nv
    sample2   = np.asarray(sample2)
    nsamples2 = sample2.shape[0]
    viss   = sample2[:, :nvis2] # samples of visible units
    hids   = sample2[:, nvis2:] # samples of hidden units

    allv2,allf1 = [],[]

    # average visible firing rates
    vv = np.mean(viss,axis=0)

    # average visible activations
    va = finv(vv)
    
    if mask is None:
        for ih in range(nh): # for each hidden unit
            hs = hids[:,ih] # look at all its samples
            h0 = np.where(hs==0)[0] # save indices of the samples where neuron was silent
            h1 = np.where(hs==1)[0] # save indices of the samples where neuron was firing
            ph = np.mean(hs) # get mean activation across all samples
            v1 = np.mean(viss[h1,:],axis=0) 
            v2 = f(va+weights[:,ih]*(1-ph)) 
            allv2.extend(ph*v2)
            allf1.extend(np.var(hs[:,None]*viss,axis=0, ddof=1))

    else: 
        # compute estimates only for active weights
        for ih in range(nh): # for each hidden unit
            hs = hids[:,ih] # look at all its samples
            h0 = np.where(hs==0)[0] # save indices of the samples where neuron was silent
            h1 = np.where(hs==1)[0] # save indices of the samples where neuron was firing
            ph = np.mean(hs) # get mean activation across all samples
            v1 = np.mean(viss[h1,:],axis=0) 
            v2 = f(va[np.where(mask[:,ih] != 0)]+weights[:,ih][np.where(mask[:,ih] != 0)]*(1-ph)) 
            allv2.extend(ph*v2)
            allf1.extend(np.var(hs[:,None]*viss,axis=0, ddof=1)[np.where(mask[:,ih] != 0)])
            
    allv2 = np.array(allv2)
    allf1 = np.array(allf1)
    allf2 = allv2*(1-allv2)
    
    return allf1, allf2 # allf1 = variance estimate, allf2 = heuristic estimate

