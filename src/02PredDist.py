import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torch.utils.data import random_split
import time

n_epochs = 10
n_job = 2

start = time.time()
print("Starting training script...") # test

train_seq = pd.read_csv("../data/train_sequences.csv")
train_lbl = pd.read_csv("../data/train_labels.csv")

# train_seq = pd.read_csv("../toy_data/train_sequences.csv")
# train_lbl = pd.read_csv("../toy_data/train_labels.csv")

train_lbl = train_lbl.infer_objects()
train_lbl = train_lbl.interpolate() # For now, interpolate - are there better imputation techniques?

train_lbl["ID_num"] = [n+1 for n in range(len(train_lbl))] # map ID to numeric ID to store in tensor
id_mapping = {idx+1: og_id for idx, og_id in enumerate(train_lbl['ID'])} # create mapping to re-map back to original ID
id_mapping[0] = "padded_row"

train_lbl[train_lbl.iloc[:,3:6].isna().any(axis=1)] # Check which rows  rows have NaN

# Create Dataset & Dataloader

def collate(batch):
    xs, ys, ids = zip(*batch)
    len_x = [x.size(0) for x in xs]

    x_padded = pad_sequence(xs, batch_first=True)
    y_padded = pad_sequence(ys, batch_first=True)
    id_padded = pad_sequence(ids, batch_first=True)

    return x_padded, y_padded, id_padded, torch.tensor(len_x)


nts = ['G', 'U', 'C', 'A', 'X', '-']
mapping = {nt: idx+1 for idx, nt in enumerate(nts)}
reverse_mapping = {v: k for k, v in mapping.items()}


def tokenise_seq(seq, mapping=mapping):
    seq_idx = [mapping[nt] for nt in seq]
    seq_idx = torch.tensor(seq_idx)
    return seq_idx

def make_coord_tensor(train_lbl):
    train_lbl['base_ID'] = train_lbl['ID'].str.rsplit('_', n=1).str[0] # sequence ID for each nt
    main_id_list = train_lbl['ID']
    y_list = []
    og_id_list_temp = [] # not extended list
    for idx in list(train_lbl['base_ID'].unique()):
        subset = train_lbl[train_lbl['base_ID'] == idx]
        coords = []
        for res in range(len(subset['ID'])):
            coord = list(subset.iloc[res, 3:6])
            coords.append(coord)
        
        og_id_list_temp.append(torch.tensor(list(subset['ID_num'])))
        
        y_list.append(torch.tensor(coords, dtype=torch.float32))
        
    #y_tensor = pad_sequence(y_list, batch_first=True)
    og_id_list = pad_sequence(og_id_list_temp, batch_first=True)

    return y_list, og_id_list

class Rnadataset(Dataset):
    def __init__(self, train_seq, train_lbl):
        super().__init__()
        self.X_list = [tokenise_seq(seq) for seq in train_seq['sequence']]
        #self.X_tensor = pad_sequence(self.X_list, batch_first=True)
        
        self.y_list, self.ids = make_coord_tensor(train_lbl)
        if all(train_lbl["base_ID"].unique() == train_seq['target_id']): # Always good to check
            print("Order corresponds between sequences and coordinates")
        else:
            raise ValueError("Mismatch between base_IDs in train_lbl and target_ids in train_seq.")
            
        #self.ids = train_seq['target_id']

    def __len__(self):
        return len(self.X_list)
    
    def __getitem__(self, index) :
        return self.X_list[index], self.y_list[index], self.ids[index]
    
dataset = Rnadataset(train_seq, train_lbl)

train_size = int(len(dataset)*0.8)
test_size = int(len(dataset)-train_size)

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=False, collate_fn=collate, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate, num_workers=8, pin_memory=True)

# Map coord to d & vice versa

def pairwise_distance_matrix(X):
    diff = X.unsqueeze(2) - X.unsqueeze(1)  # shape: (batch, 35, 35, 5)
    return torch.norm(diff, dim=-1)

def distances_to_coords(D):
    L = D.shape[0] # D: (L, L) symmetric, zero diagonal
    I = torch.eye(L) # Centering matrix - LxL identity
    ones = torch.ones((L, L)) / L
    H = I - ones

    D2 = D**2 # Squared distances
    B = -0.5 * H @ D2 @ H # Double‐centered Gram matrix

    # Eigen‐decomposition
    eigvals, eigvecs = torch.linalg.eigh(B)
    # Sort descending
    idx = torch.argsort(eigvals, descending=True)
    vals = eigvals[idx][:3]
    vecs = eigvecs[:, idx][:, :3]

    # Coordinates = V * sqrt(Λ)
    return vecs * torch.sqrt(vals).unsqueeze(0)

# Define blocks of the model

class SeqEncoder(nn.Module): # Define single encoder block
    def __init__(self, hidden_size=128, kernel_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding = kernel_size // 2)
        self.attn = nn.MultiheadAttention(hidden_size, 8)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.GELU(),
            nn.Linear(4*hidden_size, hidden_size)
        )

    def forward(self, X, padding_mask=None):
        X = X + self.conv(X.transpose(1,2)).transpose(1,2) # 1D conv with residual connection + Layer Norm; transpose to expected input
        X = self.norm1(X)
        res = X
        attn_out, _ = self.attn(X.transpose(0,1), X.transpose(0,1), X.transpose(0,1), key_padding_mask=padding_mask)
        attn_out = attn_out.transpose(0,1) + res
        X = self.norm2(attn_out)
        res = X
        X = self.norm3(res + self.ff(X))
        return X
        
class ConvEncoder(nn.Module): # define a whole transformer pipeline
    def __init__(self, n_blocks = 9, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([SeqEncoder(**kwargs) for _ in range(n_blocks)])
    
    def forward(self, X, padding_mask=None):
        for layer in self.layers:
            X = layer(X, padding_mask=padding_mask)
        return X

class DistancePredictor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        f = 4 * hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(f, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1))
        
    def forward(self, X):
        b, l, d = X.size()
        Xi = X.unsqueeze(2).expand(-1, -1, l, -1) # position i
        Xj = X.unsqueeze(1).expand(-1, l, -1, -1) # position j
        f = torch.cat([Xi, Xj, Xi-Xj, Xi*Xj], dim = -1) # stack i & j repr, their distance (-), and similarity (*)
        d = self.mlp(f).squeeze(-1)
        d = torch.relu(d)
        d = (d+d.transpose(1,2))*0.5 # symmetric
        d = d.masked_fill(torch.eye(l).bool(), 0.) # 0 across diagonal
        return d   

# Define model 

class InitModel(Module): # define rest of model
    def __init__(self, vocab=6, max_len = 1024, n_blocks=9, hidden_size=256):
        super().__init__()
        self.b = vocab
        self.embedding = nn.Embedding(self.b, hidden_size, padding_idx=0) # map each base to a vector representation of size 256
        self.pos_embedding = nn.Embedding( max_len, hidden_size)
        self.convencoder = ConvEncoder(n_blocks=n_blocks, hidden_size=hidden_size)
        self.output=DistancePredictor(hidden_size=hidden_size)
        #self.output = nn.Linear(hidden_size, 3)

    def forward(self, X):

        # Make embeddings (+ positional embeddings)

        pad_mask = (X == 0)
        seq_length = X.size()[1]

        X = self.embedding(X)
        positions = torch.arange(seq_length).unsqueeze(0).expand(X.size(0), seq_length)
        pos_embd = self.pos_embedding(positions)
        X = X + pos_embd

        # Pass through convolutional transformer

        X = self.convencoder(X, padding_mask=pad_mask)

        out = self.output(X)
        return(out)


# TRAIN 
max_seq_len = max(x.size(0) for x in dataset.X_list) * 2

initmodel = InitModel()
criterion = MSELoss()
optimiser = Adam(initmodel.parameters())

cols = ["Epoch", "Train_Loss", "Test_Loss"]
perf = pd.DataFrame(index=range(n_epochs), columns=cols)

for epoch in range(n_epochs):
    loss_train = []
    epoch_pred_train = []
    epoch_true_train = []
    num_ids = []
    seq_idx = []
    initmodel.train()
    for seq, coords, ids, seq_lens in train_loader:
        pad_mask = (seq == 0)
        optimiser.zero_grad()
        true = pairwise_distance_matrix(coords)
        pred_i = initmodel(seq)
        mask = (seq!=0).unsqueeze(1).expand_as(pred_i)
        pred = pred_i[mask]
        true = true[mask]
        loss = criterion(pred,true)
        loss_train.append(loss.item())
        num_ids.extend(ids.flatten(0,1).tolist())
        loss.backward()
        optimiser.step()

    loss_train = sum(loss_train)/len(loss_train)

    initmodel.eval()
    with torch.no_grad():
        loss_test = []
        for seq, coords, ids, seq_lens in test_loader:
            pred_test = initmodel(seq)
            true_test = pairwise_distance_matrix(coords)
            mask = (seq!=0).unsqueeze(1).expand_as(pred_test)
            pred_test = pred_test[mask]
            true_test = true_test[mask]
            loss = criterion(pred, true)
            loss_test.append(loss.item())
        
        loss_test_val = sum(loss_test)/len(loss_test)
    
    perf.iloc[epoch, :] = [epoch+1, loss_train, loss_test]
    print(f"Epoch {epoch+1}: Loss train {round(loss_train, 2)}, Loss Test {round(loss_test_val, 2)}")

perf.to_csv("../outputs/DistPred/distpred_perf.csv")

end = time.time()

print(f"Time: {end-start}s")

# VALIDATE

# Get validation set

validation_seq = pd.read_csv("../data/validation_sequences.csv")
validation_lbl = pd.read_csv("../data/validation_labels.csv")
validation_lbl["ID_num"] = [n+1 for n in range(len(validation_lbl))] # map ID to numeric ID to store in tensor
id_mapping_val = {idx+1: og_id for idx, og_id in enumerate(validation_lbl['ID'])} # create mapping to re-map back to original ID
id_mapping_val[0] = "padded_row"
val_set = Rnadataset(validation_seq, validation_lbl)
val_loader = DataLoader(val_set,batch_size=32,shuffle=False,num_workers=4,pin_memory=False,collate_fn=collate)

# Make predictions on validation set
initmodel.eval()
all_preds = []
with torch.no_grad():
    for seq, coords, ids, lengths in val_loader:
        pred = initmodel(seq)         
        for b in range(pred.size(0)):
            single = pred[b]          
            mask   = seq[b] != 0
            coords = distances_to_coords(single)
            coords = coords[mask]
            all_preds.append(coords)

stacked = torch.cat(all_preds, dim=0)

submission_cols = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2',
       'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']

submission_df = pd.DataFrame(0.0, index = range(stacked.size()[0]), columns = submission_cols)
submission_df[['ID', 'resname', 'resid']] = validation_lbl[['ID', 'resname', 'resid']]

submission_df[['x_1', 'y_1', 'z_1']] = stacked.detach().numpy()
submission_df.dtypes

submission_df.to_csv("../outputs/DistPred/submission2.csv")

