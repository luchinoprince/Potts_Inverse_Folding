import torch

"""
First line:

$A^{b,m}_{ia} = \sum_{kc} \delta^{b,m}_{kc} J^{b}_{kc, ia} + h^{b}_{ia}$

Second line:

$Z^{b,m}_i = \log \sum_{a} \exp \left (\sum_{kc} \delta^{b,m}_{kc} J^b_{kc, ia} + h^{b}_{ia} \right)$

Third line:

$C^{b,m}_{i} = \sum_{a} \sum_{kc} \delta^{b,m}_{kc} J^{b}_{kc, ia} \delta^{b,m}_{ia} + h^{b}_{ia} \delta^{b,m}_{ia}$

Fourth Line:

$D^{b,m}_{i} = \sum_{a} \sum_{kc} \delta^{b,m}_{kc} J^{b}_{kc, ia} \delta^{b,m}_{ia} + h^{b}_{ia} \delta^{b,m}_{ia} - \log \sum_{a} \exp \left (\sum_{kc} \delta^{b,m}_{kc} J^b_{kc, ia} + h^{b}_{ia} \right)$

Fourth line with $\delta$ resolved:

$D^{b,m}_{i}  = \sum_{k} J^b_{k,i}(s^{b,m}_k, s^{b,m}_i) + h_i(s^{b,m}_i) - \log \sum_{a} \exp \left( J^b_{k,i}(s^{b,m}_k, a) + h^b_i(a) \right)$

"""


def get_npll(msas_embedded, couplings, fields, N, q):
    """ Get negative pseudo log likelihood (npll)
    """
    B, M, _ = msas_embedded.shape
    ## The other value would be the length of the proteins in MSAs
    # (B, M, N*q) x (B, N*q, N*q) + (B, 1, N*q) -> (B, M, N*q) -> (B, M, N, q)
    A = (msas_embedded @ couplings + fields.unsqueeze(1)).view(B, M, N, q)

    # (B, M, N, q) -> (B, M, N)
    Z = torch.logsumexp(A, dim=-1)

    # (B, M, N, q) * (B, M, N, q) -> (B, M, N, q) -> (B, M, N)
    C = torch.sum(A * msas_embedded.view(B, M, N, q), dim=-1)

    # (B, M, N) - (B, M, N) -> (B, M, N)
    pll = C - Z

    return -pll#, C, Z

def get_npll_indep(msas_embedded, F, N, q):
    ## V and F have already been padded
    ## msas_embedded: (B, M, N, q)
    ## F: (B, N, q) of fields. I have already summed over the K-th dimension. Fields DON'T need re-engeneering
    ## inv_padding_mask: (B, N)
    B, M, _, _ = msas_embedded.shape
    
    
    ## (B, 1, N, q) * (B, M, N, q) = (B, M, N, q) ---> (B, M, N)
    Fi_data = torch.sum(torch.unsqueeze(F, dim=1) * msas_embedded, axis=-1)
    
    ## Fs is the field componenet of the sequence s: (B, M, N) --> (B, M) 
    Fs = torch.sum(Fi_data, axis=-1)
    
    ## Fields of the sequence having amino acid substitution: 
    Fc = (Fs.unsqueeze(axis=-1).unsqueeze(axis=-1) - Fi_data.unsqueeze(axis=-1) + F.unsqueeze(dim=1))
    
    
    ## (B, M, 1) - (B, M, N) = (B, M, N)
    pll = torch.unsqueeze(Fs, axis=-1) - torch.logsumexp(Fc, axis=-1)
    
    return -pll#, E, torch.logsumexp(Ec, axis=-1)    


def get_npll2(msas_embedded, V, F, N, q):
    ## V and F have already been padded
    ## msas_embedded: (B, M, N, q)
    ## V: (B, K, N, q)
    ## F: (B, N, q) of fields. I have already summed over the K-th dimension. Fields DON'T need re-engeneering
    ## inv_padding_mask: (B, N)
    K = V.shape[1]
    B, M, _, _ = msas_embedded.shape
    
    
    ## (B, 1, N, q) * (B, M, N, q) = (B, M, N, q) ---> (B, M, N)
    Fi_data = torch.sum(torch.unsqueeze(F, dim=1) * msas_embedded, axis=-1)
    
    ## Fs is the field componenet of the sequence s: (B, M, N) --> (B, M) 
    Fs = torch.sum(Fi_data, axis=-1)
    
    ## Fields of the sequence having amino acid substitution: 
    Fc = (Fs.unsqueeze(axis=-1).unsqueeze(axis=-1) - Fi_data.unsqueeze(axis=-1) + F.unsqueeze(dim=1))
    

    ## (B, 1, K, M, q) * (B, M, 1, N, q) = (B, M, K, N, q)  ---> (B, M, K, N)
    V_data = torch.sum(torch.unsqueeze(V, dim=1) * 
                       torch.unsqueeze(msas_embedded, dim=2), axis=-1)
    
    ## (B, M, K, N)  ---> (B, M, K)
    S_k = torch.sum(V_data, axis=-1)
    
    ## (B, M, K, N)  ---> (B, M, K)
    H_k = torch.sum(V_data**2, axis=-1)
    
    ## (B, M, K)  ---> (B, M)
    E = torch.sum(S_k**2 - H_k, axis=-1)
    
    ## (B, M, K, 1, 1) - (B, M, K, N, 1) + (B, 1, K, N, q) = (B, M, K, N ,q)
    Sc_k = (S_k.unsqueeze(-1).unsqueeze(-1) - V_data.unsqueeze(-1) + V.unsqueeze(1))
    
    ## (B, M, K, 1, 1) - (B, M, K, N, 1) + (B, 1, K, N, q) = (B, M, K, N ,q)
    Hc_k = (H_k.unsqueeze(-1).unsqueeze(-1) - V_data.unsqueeze(-1)**2 + V.unsqueeze(1)**2)
    
    ## (B, M, K, N ,q) - (B, M, K, N ,q) = (B, M, K, N ,q) ---> (B, M, N, q)
    Ec = torch.sum(Sc_k**2 - Hc_k, axis=2)
    
    ## (B, M, 1) - (B, M, N) = (B, M, N)
    pll = torch.unsqueeze(0.5*E + Fs, axis=-1) - torch.logsumexp(0.5*Ec + Fc, axis=-1)
    
    return -pll #, E, torch.logsumexp(Ec, axis=-1)    ## this should return the different log predictives


def get_npll3(msas_embedded, couplings, fields, N, q):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    localfiels = torch.einsum("bNP,bmP->bmN", [couplings,msas_embedded] )
    localfiels = localfiels + fields.unsqueeze(1)
    localfiels = localfiels.reshape((-1, q))
    msas_embedded= msas_embedded.reshape((-1, q))
    pll = loss(localfiels.reshape((-1, q)), msas_embedded.argmax(dim=-1).flatten())
    #print(pll)
    #print(pll.shape)
    return pll










































def get_npll_new(msas_embedded, param_embeddings, fields, N, q):
    ## MSAS EMBEDDED is the matrix with ones and zeros telling the 
    ## identity of the amino acids at different positions
    ## (B, M, N*q) x (B, N*q, n_param_heads) = (B, M, n_param_heads)
    B, M, _ = msas_embedded.shape
    n_param_heads = param_embeddings.shape[1]
    K = n_param_heads
    
    ## (B,M,K,N*q)
    aux = torch.mul(msas_embedded.unsqueeze(2), param_embeddings.unsqueeze(1))
    
    ## E_full gives the sum of all the v_ik delta_ik for the true sequence: (B,M,K)
    E_full = torch.sum(aux, axis=-1)            #msas_embedded @ torch.transpose(param_embeddings, 1, 2)
    
    ## Extra gives the sum of (v_ik delta_ik)**2 for the true sequence: (B,M,K)
    extra = torch.sum(aux**2, axis=-1)
    
    ## E_sub gives the true components of the latent vectors for the true sequence: (B,M,K,N)    
    E_sub  = torch.sum(torch.mul(torch.unsqueeze(msas_embedded.view(B,M,N,q), 2),
                                 torch.unsqueeze(param_embeddings.view(B,n_param_heads,N,q), 1)),
                       axis = -1)
    ## param embedding is (B, M, N, q)
    
    
    ## E_aux should be (B, M, K, N, q)    
    # E_minus gives us the sums v_ik delta_ik other that the i-th component for all i: (B, M, K, N, 1)
    E_minus_i = E_full.unsqueeze(-1).unsqueeze(-1) - torch.unsqueeze(E_sub, -1)
    
    #E_minus_2 gives us the sums (v_ik delta_ik)^2 other that the i-th component for all i: (B, M, K, N, 1)
    E_minus_i_2 = (extra.unsqueeze(-1) - E_sub**2).unsqueeze(-1)
    
    ##E_aux gives the energy for E_c for all i, c: (B, M, K, N, q)! 
    ## Maybe error is here.... check suggests that it is correct
    E_c  = (E_minus_i + 
        torch.unsqueeze(param_embeddings.view(B,K,N,q), 1))
    
    ## Extra_c does the same of extra but for E_c for different i and c: (B,M,K,N,q)
    extra_c = (E_minus_i_2 + 
                torch.unsqueeze(param_embeddings.view(B,K,N,q), 1)**2)
    
    ## E_terms is (B, M, N, q), we sum across param_embeddings
    E_terms = torch.sum(E_c**2 - extra_c, dim=2)  
    ## First term of next sum is (B, M, 1) second is (B, M, N) -> B, M, N
    C = torch.sum(E_full**2 - extra, dim=2)
    pll = torch.unsqueeze(C, dim=-1) - torch.logsumexp(E_terms, axis=-1) ##this sum is the denominator of the predictive
    
    return -pll, C, torch.logsumexp(E_terms, axis=-1)   ##this should give the same output so then the loss should be the same