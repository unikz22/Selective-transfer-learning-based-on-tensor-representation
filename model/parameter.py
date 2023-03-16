import torch

def get_vseq(FM,seq1,seq2,seq3,seq4,seq):
    seq1_v=FM[0]
    seq2_v=FM[1]
    seq3_v=FM[2]
    seq4_v=FM[3]
    
    a = seq1['lstm.weight_ih_l0']*seq1_v+seq2['lstm.weight_ih_l0']*seq2_v+seq3['lstm.weight_ih_l0']*seq3_v+seq4['lstm.weight_ih_l0']*seq4_v
    b = seq1['lstm.weight_hh_l0']*seq1_v+seq2['lstm.weight_hh_l0']*seq2_v+seq3['lstm.weight_hh_l0']*seq3_v+seq4['lstm.weight_hh_l0']*seq4_v
    c = seq1['lstm.bias_ih_l0']*seq1_v+seq2['lstm.bias_ih_l0']*seq2_v+seq3['lstm.bias_ih_l0']*seq3_v+seq4['lstm.bias_ih_l0']*seq4_v
    d = seq1['lstm.bias_hh_l0']*seq1_v+seq2['lstm.bias_hh_l0']*seq2_v+seq3['lstm.bias_hh_l0']*seq3_v+seq4['lstm.bias_hh_l0']*seq4_v
    
    e = seq1['lstm.weight_ih_l1']*seq1_v+seq2['lstm.weight_ih_l1']*seq2_v+seq3['lstm.weight_ih_l1']*seq3_v+seq4['lstm.weight_ih_l1']*seq4_v
    f = seq1['lstm.weight_hh_l1']*seq1_v+seq2['lstm.weight_hh_l1']*seq2_v+seq3['lstm.weight_hh_l1']*seq3_v+seq4['lstm.weight_hh_l1']*seq4_v
    g = seq1['lstm.bias_ih_l1']*seq1_v+seq2['lstm.bias_ih_l1']*seq2_v+seq3['lstm.bias_ih_l1']*seq3_v+seq4['lstm.bias_ih_l1']*seq4_v
    h = seq1['lstm.bias_hh_l1']*seq1_v+seq2['lstm.bias_hh_l1']*seq2_v+seq3['lstm.bias_hh_l1']*seq3_v+seq4['lstm.bias_hh_l1']*seq4_v
    
    i = seq1['lstm.weight_ih_l2']*seq1_v+seq2['lstm.weight_ih_l2']*seq2_v+seq3['lstm.weight_ih_l2']*seq3_v+seq4['lstm.weight_ih_l2']*seq4_v
    j = seq1['lstm.weight_hh_l2']*seq1_v+seq2['lstm.weight_hh_l2']*seq2_v+seq3['lstm.weight_hh_l2']*seq3_v+seq4['lstm.weight_hh_l2']*seq4_v
    k = seq1['lstm.bias_ih_l2']*seq1_v+seq2['lstm.bias_ih_l2']*seq2_v+seq3['lstm.bias_ih_l2']*seq3_v+seq4['lstm.bias_ih_l2']*seq4_v
    l = seq1['lstm.bias_hh_l2']*seq1_v+seq2['lstm.bias_hh_l2']*seq2_v+seq3['lstm.bias_hh_l2']*seq3_v+seq4['lstm.bias_hh_l2']*seq4_v
    
    m = seq1['linear_Layer.0.weight']*seq1_v+seq2['linear_Layer.0.weight']*seq2_v+seq3['linear_Layer.0.weight']*seq3_v+seq4['linear_Layer.0.weight']*seq4_v
    n = seq1['linear_Layer.0.bias']*seq1_v+seq2['linear_Layer.0.bias']*seq2_v+seq3['linear_Layer.0.bias']*seq3_v+seq4['linear_Layer.0.bias']*seq4_v
    o = seq1['linear_Layer.1.weight']*seq1_v+seq2['linear_Layer.1.weight']*seq2_v+seq3['linear_Layer.1.weight']*seq3_v+seq4['linear_Layer.1.weight']*seq4_v
    p = seq1['linear_Layer.1.bias']*seq1_v+seq2['linear_Layer.1.bias']*seq2_v+seq3['linear_Layer.1.bias']*seq3_v+seq4['linear_Layer.1.bias']*seq4_v
    q = seq1['linear_Layer.3.weight']*seq1_v+seq2['linear_Layer.3.weight']*seq2_v+seq3['linear_Layer.3.weight']*seq3_v+seq4['linear_Layer.3.weight']*seq4_v
    r = seq1['linear_Layer.3.bias']*seq1_v+seq2['linear_Layer.3.bias']*seq2_v+seq3['linear_Layer.3.bias']*seq3_v+seq4['linear_Layer.3.bias']*seq4_v
    s = seq1['linear_Layer.4.weight']*seq1_v+seq2['linear_Layer.4.weight']*seq2_v+seq3['linear_Layer.4.weight']*seq3_v+seq4['linear_Layer.4.weight']*seq4_v
    t = seq1['linear_Layer.4.bias']*seq1_v+seq2['linear_Layer.4.bias']*seq2_v+seq3['linear_Layer.4.bias']*seq3_v+seq4['linear_Layer.4.bias']*seq4_v
    u = seq1['linear_out.weight']*seq1_v+seq2['linear_out.weight']*seq2_v+seq3['linear_out.weight']*seq3_v+seq4['linear_out.weight']*seq4_v
    v = seq1['linear_out.bias']*seq1_v+seq2['linear_out.bias']*seq2_v+seq3['linear_out.bias']*seq3_v+seq4['linear_out.bias']*seq4_v
    
    with torch.no_grad():
        for name, param in seq.named_parameters():
            if 'lstm.weight_ih_l0' in name:
                param.copy_(a)
            elif 'lstm.weight_hh_l0' in name:
                param.copy_(b)
            elif 'lstm.bias_ih_l0' in name:
                param.copy_(c)
            elif 'lstm.bias_hh_l0' in name:
                param.copy_(d)
            elif 'lstm.weight_ih_l1' in name:
                param.copy_(e)
            elif 'lstm.weight_hh_l1' in name:
                param.copy_(f)
            elif 'lstm.bias_ih_l1' in name:
                param.copy_(g)
            elif 'lstm.bias_hh_l1' in name:
                param.copy_(h)
            elif 'lstm.weight_ih_l2' in name:
                param.copy_(i)
            elif 'lstm.weight_hh_l2' in name:
                param.copy_(j)
            elif 'lstm.bias_ih_l2' in name:
                param.copy_(k)
            elif 'lstm.bias_hh_l2' in name:
                param.copy_(l)
            elif 'linear_Layer.0.weight' in name:
                param.copy_(m)
            elif 'linear_Layer.0.bias' in name:
                param.copy_(n)
            elif 'linear_Layer.1.weight' in name:
                param.copy_(o)
            elif 'linear_Layer.1.bias' in name:
                param.copy_(p)
            elif 'linear_Layer.3.weight' in name:
                param.copy_(q)
            elif 'linear_Layer.3.bias' in name:
                param.copy_(r)
            elif 'linear_Layer.4.weight' in name:
                param.copy_(s)
            elif 'linear_Layer.4.bias' in name:
                param.copy_(t)
            elif 'linear_out.weight' in name:
                param.copy_(u)
            elif 'linear_out.bias' in name:
                param.copy_(v)
    
    if (max(FM)-min(FM))<0.25:
        flag=4
    else:
        flag=8
    i=0
    for para in seq.lstm.parameters():
        i = i+1
        para.requires_grad = False
        if i == flag:
            break
    return seq