# # X: One-hot encode sequence and convert to tenshor

# #seq = train_seq['sequence'][0] # to test

# nts = ['G', 'U', 'C', 'A', 'X', '-']
# mapping = {nt: idx for idx, nt in enumerate(nts)}

# def one_hot_encode_seq(seq):

#     ohe_seq = []

#     for nt in seq:
#         binary_l = [0] * len(nts)
#         binary_l[mapping[nt]] = 1
#         ohe_seq.append(binary_l)
    
#     ohe_torch = torch.tensor(ohe_seq, dtype=torch.float32)
#     return ohe_torch

# X_list = [one_hot_encode_seq(seq) for seq in train_seq['sequence']]
# X_tensor = pad_sequence(X_list, batch_first=True) # pad sequences to same length