import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, MSELoss
from torch.optim import Adam
from torch.utils.data import random_split
import time

n_epochs = 1
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

# Define blocks of the model

class SeqEncoder(nn.Module): # Define single encoder block
    def __init__(self, hidden_size=256, kernel_size=3):
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
    

# Define model 

class InitModel(Module): # define rest of model
    def __init__(self, vocab=7, max_len = 1024, n_blocks=9, hidden_size=256):
        super().__init__()
        self.b = vocab
        self.embedding = nn.Embedding(self.b, hidden_size, padding_idx=0) # map each base to a vector representation of size 256
        self.pos_embedding = nn.Embedding( max_len, hidden_size)
        self.convencoder = ConvEncoder(n_blocks=n_blocks, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, 3)

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
    
# Define custom loss function on distance matrices rather than coords

def pairwise_distance_matrix(X):
    diff = X.unsqueeze(2) - X.unsqueeze(1)  # shape: (batch, 35, 35, 5)
    return torch.norm(diff, dim=-1)

class DistanceMatrixLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = MSELoss()
    
    def forward(self, y_true, y_pred, padding_mask):
        y_true_m = pairwise_distance_matrix(y_true)
        y_pred_m = pairwise_distance_matrix(y_pred)

        valid = (~padding_mask).unsqueeze(2) & (~padding_mask).unsqueeze(1)
        se = (y_true_m - y_pred_m).pow(2)
        se_valid = se[valid]
        return se_valid.mean()



# Define function to convert coordinates to dataframe for TMScore calculation
def coords_to_df_train(tensor_list):
    flat_tensor = torch.cat(tensor_list, dim=0).flatten(0,1) # fuse tensors in list, then flatten (batch + seq)

    n_seq = 0
    seq_length = tensor_list[0].size()[1]
    for i in tensor_list: # calculate number of sequences 
        n_seq = n_seq + i.size()[0] 
    
    seq_ids = torch.arange(n_seq).repeat_interleave(seq_length).unsqueeze(1) # create ID for each seq in flat tensor
    pred_idxs = torch.cat([seq_ids, flat_tensor], dim=1) # fuse IDs with tensor itself
    df = pd.DataFrame(pred_idxs.detach().numpy()) # convert to dataframe
    df.columns = ['seq_ID_int', "x", "y", "z"] 
    return df 

# TRAIN 
max_seq_len = max(x.size(0) for x in dataset.X_list) * 2

initmodel = InitModel(max_len=max_seq_len)
criterion = DistanceMatrixLoss()
optimiser = Adam(initmodel.parameters())

cols = ["Epoch", "Train_Loss", "Test_Loss"]
perf = pd.DataFrame(index=range(n_epochs), columns=cols)

for epoch in range(n_epochs):
    print(f"epoch {epoch+1}")
    loss_train = []
    num_ids = []
    seq_idx = []
    initmodel.train()
    for seq, coords, ids, lengths in train_loader:
        pad_mask = (seq == 0)
        optimiser.zero_grad()
        pred_coords = initmodel(seq)
        loss = criterion(coords,pred_coords, pad_mask)
        loss_train.append(loss.item())
        num_ids.extend(ids.flatten(0,1).tolist())
        seq_idx.extend(seq.flatten(0,1).tolist())
        loss.backward()
        optimiser.step()

    loss_train = sum(loss_train)/len(loss_train)

    initmodel.eval()
    with torch.no_grad():
        loss_test = []
        for seq, coords, ids, lengths in test_loader:
            pad_mask = (seq == 0)
            pred_coords_test = initmodel(seq)
            loss = criterion(coords, pred_coords_test, pad_mask)
            loss_test.append(loss.item())
        
        loss_test_val = sum(loss_test)/len(loss_test)
    
    perf.iloc[epoch, :] = [epoch+1, loss_train, loss_test_val]
    print(f"Epoch {epoch+1}: Loss train {round(loss_train, 2)}, Loss Test {round(loss_test_val, 2)}")

perf.to_csv(f'../outputs/InitialModel/initialmodel_perf{n_job}.csv')

end = time.time()
print(f"Time for job: {end-start}s")

# VALIDATE

# Get validation set

validation_seq = pd.read_csv("../data/validation_sequences.csv")
validation_lbl = pd.read_csv("../data/validation_labels.csv")
validation_lbl["ID_num"] = [n+1 for n in range(len(validation_lbl))] # map ID to numeric ID to store in tensor
id_mapping_val = {idx+1: og_id for idx, og_id in enumerate(validation_lbl['ID'])} # create mapping to re-map back to original ID
id_mapping_val[0] = "padded_row"
val_set = Rnadataset(validation_seq, validation_lbl)

# Make predictions on validation set
initmodel.eval()
val_pred = initmodel(val_set.X_tensor)

mask_val = (val_set.X_tensor.flatten(0,1) != 0)
val_pred_flat = val_pred.flatten(0,1)
val_seq_pred = val_pred_flat[mask_val]

submission_cols = ['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2',
       'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']

submission_df = pd.DataFrame(0.0, index = range(val_seq_pred.shape[0]), columns = submission_cols)
submission_df[['ID', 'resname', 'resid']] = validation_lbl[['ID', 'resname', 'resid']]
submission_df[['x_1', 'y_1', 'z_1']] = val_seq_pred.detach().numpy()
submission_df.dtypes
submission_df.to_csv(f'../outputs/InitialModel/submission{n_job}.csv')

