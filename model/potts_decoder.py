import torch
from torch.nn import Linear, TransformerEncoderLayer, LayerNorm
from torchvision.ops import MLP
import numpy as np

from torch.nn.functional import softmax
from torch.distributions import Categorical

class PottsDecoder(torch.nn.Module):

    def __init__(self, q, n_layers, d_model, input_encoding_dim, param_embed_dim, n_heads, n_param_heads, dropout=0.0):

        super().__init__()
        self.q = q
        self.n_layers = n_layers
        self.d_model = d_model
        self.input_encoding_dim = input_encoding_dim
        self.param_embed_dim = param_embed_dim
        self.n_heads = n_heads
        self.n_param_heads = n_param_heads
        self.dropout = dropout

        self.input_MLP = Linear(self.input_encoding_dim, self.d_model)


        self.attention_layers = torch.nn.ModuleList([])
        self.relu = torch.nn.ReLU()
        for _ in range(n_layers):
            attention_layer = TransformerEncoderLayer(self.d_model, self.n_heads,
                                                      dropout=self.dropout, batch_first=True)
            self.attention_layers.append(attention_layer)


        
        self.P = Linear(self.d_model, self.n_param_heads*self.d_model, bias=False)   ## this uses a more sensible initialization
        self.output_linear = Linear(self.d_model, self.q)

        self.field_linear = Linear(self.q, self.q)

    def _get_params(self, param_embeddings, N, padding_mask):
        
        padding_mask_inv = (~padding_mask)

        # set embeddings to zero where padding is present
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3)

        # get fields ---> here I sum over K!
        fields = torch.sum(self.field_linear(param_embeddings), dim=1)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        # flatten fields
        fields = fields.view(-1, N * self.q)

        # flatten to (B, n_param_heads, N*q)
        param_embeddings = param_embeddings.flatten(start_dim=2, end_dim=3)

        # outer to (B, N*q, N*q)
        couplings = torch.einsum('bpi, bpj -> bij', (param_embeddings, param_embeddings))

        # create mask for couplings
        t = torch.ones(self.q, self.q)
        mask_couplings = (1 - torch.block_diag(*([t] * N))).to(couplings.device)
        mask_couplings.requires_grad = False

        couplings = couplings * mask_couplings

        return couplings/np.sqrt(self.n_param_heads), fields/np.sqrt(self.n_param_heads)
    
    def forward(self, encodings, padding_mask):

        B, N, _ = encodings.shape

        assert B == padding_mask.shape[0]
        assert N == padding_mask.shape[1]
        
        #with profiler.record_function("Embeddings"):
        embeddings = self.input_MLP(encodings)
        #with profiler.record_function("Attention Layers"):
        for attention_layer in self.attention_layers:
            embeddings = attention_layer(embeddings, src_key_padding_mask=padding_mask)
            embeddings = self.relu(embeddings)

        param_embeddings = torch.transpose(self.P(embeddings).reshape(B, N, self.n_param_heads, self.d_model), 1, 2)

        # apply relu
        param_embeddings = self.relu(param_embeddings)

        # (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        couplings, fields = self._get_params(param_embeddings, N, padding_mask)

        return couplings, fields
    
    def _get_params_new(self, param_embeddings, N, padding_mask):
        
        padding_mask_inv = (~padding_mask)

        # set embeddings to zero where padding is present
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3) 

        # get fields
        fields = torch.sum(self.field_linear(param_embeddings), dim=1) *  self.n_param_heads**(-1/2)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        ## To normalize later computations
        param_embeddings = param_embeddings * self.n_param_heads**(-1/4)



        return param_embeddings, fields


    
    def forward_new(self, encodings, padding_mask):

        B, N, _ = encodings.shape

        assert B == padding_mask.shape[0]
        assert N == padding_mask.shape[1]

        embeddings = self.input_MLP(encodings)
        for attention_layer in self.attention_layers:
            embeddings = attention_layer(embeddings, src_key_padding_mask=padding_mask)
            embeddings = self.relu(embeddings)

        param_embeddings = torch.transpose(self.P(embeddings).reshape(B, N, self.n_param_heads, self.d_model), 1, 2) #@ embeddings.unsqueeze(1).unsqueeze(4)
        # (1, n_param_heads, 1, d_model, d_model) x (B, 1, N, d_model, 1) -> (B, n_param_heads, N, d_model)
        param_embeddings = self.relu(param_embeddings)

        # (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        param_embeddings, fields = self._get_params_new(param_embeddings, N, padding_mask)

        return param_embeddings, fields

    def _get_params_indep(self, param_embeddings, N, padding_mask):
        """ This is the function that forwards fot the model without couplings"""
        padding_mask_inv = (~padding_mask)

        # set embeddings to zero where padding is present
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3) 

        # get fields
        fields = torch.sum(self.field_linear(param_embeddings), dim=1) *  self.n_param_heads**(-1/2)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        return fields


    
    def forward_indep(self, encodings, padding_mask):
        """ This is the forward function for the model that does not have the the Couplings"""
        B, N, _ = encodings.shape

        assert B == padding_mask.shape[0]
        assert N == padding_mask.shape[1]

        embeddings = self.input_MLP(encodings)
        for attention_layer in self.attention_layers:
            embeddings = attention_layer(embeddings, src_key_padding_mask=padding_mask)
            embeddings = self.relu(embeddings)

        param_embeddings = torch.transpose(self.P(embeddings).reshape(B, N, self.n_param_heads, self.d_model), 1, 2) #@ embeddings.unsqueeze(1).unsqueeze(4)
        # (1, n_param_heads, 1, d_model, d_model) x (B, 1, N, d_model, 1) -> (B, n_param_heads, N, d_model)
        param_embeddings = self.relu(param_embeddings)

        # (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        fields = self._get_params_indep(param_embeddings, N, padding_mask)

        return fields
    
    def _get_params_ardca(self, param_embeddings, N, padding_mask):
        
        padding_mask_inv = (~padding_mask)

        # set embeddings to zero where padding is present
        param_embeddings = param_embeddings * padding_mask_inv.unsqueeze(1).unsqueeze(3)

        # get fields ---> here I sum over K!
        fields = torch.sum(self.field_linear(param_embeddings), dim=1)

        # set fields to 0 depending on the padding
        fields = fields * padding_mask_inv.unsqueeze(2)

        # flatten fields
        fields = fields.view(-1, N * self.q)

        # flatten to (B, n_param_heads, N*q)
        param_embeddings = param_embeddings.flatten(start_dim=2, end_dim=3)

        # outer to (B, N*q, N*q)
        couplings = torch.einsum('bpi, bpj -> bij', (param_embeddings, param_embeddings))

        # create mask for couplings
        t = torch.ones(self.q, self.q)
        mask_couplings = (1 - torch.block_diag(*([t] * N))).to(couplings.device)
        mask_couplings.requires_grad = False

        couplings = couplings * mask_couplings

        #### We keen only lower triangular since we want to do arDCA
        couplings = torch.tril(couplings)

        return couplings/np.sqrt(self.n_param_heads), fields/np.sqrt(self.n_param_heads)
    
    def forward_ardca(self, encodings, padding_mask):

        B, N, _ = encodings.shape

        assert B == padding_mask.shape[0]
        assert N == padding_mask.shape[1]
        
        embeddings = self.input_MLP(encodings)
        for attention_layer in self.attention_layers:
            embeddings = attention_layer(embeddings, src_key_padding_mask=padding_mask)
            embeddings = self.relu(embeddings)

        param_embeddings = torch.transpose(self.P(embeddings).reshape(B, N, self.n_param_heads, self.d_model), 1, 2)

        # apply relu
        param_embeddings = self.relu(param_embeddings)

        # (B, n_param_heads, N, q)
        param_embeddings = self.output_linear(param_embeddings)
        #with profiler.record_function("Get params"):
        couplings, fields = self._get_params_ardca(param_embeddings, N, padding_mask)

        return couplings, fields
    
    def sample_ardca(self, encodings, padding_mask, n_samples=1000):
        """Sampler for arDCA, currently works only for a single sequence.
            NB: This function should not be used for standard Potts."""
            ## Put model in evaluation mdoel
        B, N, _ = encodings.shape
        samples = torch.zeros(n_samples, N, dtype=torch.int)
        self.eval()
        q = self.q
        ## fields shape: (B,N,q), we will consider B=1 for the moment
        ## Couplings shape: (B, N*q, N*q)
        couplings, fields = self.forward_ardca(encodings, padding_mask)

        ## At the moment move to CPU! Then maybe move to GPU if we are able to vectorize
        couplings = couplings.to('cpu')
        fields = fields.to('cpu')
        ##############################################################################
        
        fields = fields[0,:].reshape(N, q)
        p_pos = softmax(-fields[0], dim=0)
        #p_pos = softmax(fields[0], dim=0)
        
        samples[:,0] = Categorical(p_pos).sample((n_samples,))
        Ham = torch.zeros(q)
        for sam in range(n_samples):
            print(f"We are at sample {sam} out of {n_samples}", end="\r")
            for pos in range(1,N):
                Ham[:] = fields[pos, :]
                for acc in range(pos):
                    for aa in range(q):
                        Ham[aa] += couplings[0, pos*q+aa, acc*q + samples[sam,acc]]
                
                p_pos[:] = softmax(-Ham, dim=0)
                #p_pos[:] = softmax(Ham, dim=0)
                samples[sam, pos] = Categorical(p_pos).sample()
        return samples
    
    def sample_ardca_full(self, encodings, padding_mask, device='cpu', n_samples=1000):
        """Sampler for arDCA, currently works for many sequences together.
            NB: This function should not be used for standard Potts."""
            ## Put model in evaluation model
        #device = 0
        ############# MOVE EVERYTHING TO CORRECT DEVICE #######
        self.eval()
        self = self.to(device)
        encodings = encodings.to(device)
        padding_mask = padding_mask.to(device)
        B, N, _ = encodings.shape
        samples = torch.zeros(n_samples, N, dtype=torch.int).to(device)
      
        q = self.q
        ## fields shape: (B,N,q), we will consider B=1 for the moment
        ## Couplings shape: (B, N*q, N*q)
        couplings, fields = self.forward_ardca(encodings, padding_mask)
        ##############################################################################
        couplings = couplings[0, :, :] #### This simplifies
        fields = fields[0,:].reshape(N, q)
        p_pos = softmax(fields[0], dim=0)
        #p_pos = softmax(-fields[0], dim=0)

        samples[:,0] = Categorical(p_pos).sample((n_samples,))

        with torch.no_grad():   ## If you take this out you will fill you RAM linearly in n_samples
            for pos in range(1,N):
                Ham = torch.zeros(n_samples, q).to(device)
                Ham += fields[pos, :]
                for acc in range(pos):
                    for aa in range(q):
                        second_idx = acc*q + samples[:, acc]
                        Ham[:, aa] += couplings[pos*q+aa, second_idx]#.unsqueeze(-1)
                p_pos = softmax(Ham, dim=1)
                #p_pos = softmax(-Ham, dim=0)
                samples[:, pos] = Categorical(p_pos).sample()
        return samples
    
    
    def sample_ardca_full_scaled(self, encodings, padding_mask, device='cpu', n_samples=1000):
        """Sampler for arDCA, currently works for many sequences together.
            NB: This function should not be used for standard Potts. Works for scaled model"""
        ############# MOVE EVERYTHING TO CORRECT DEVICE #######
        self.eval()
        self = self.to(device)
        encodings = encodings.to(device)
        padding_mask = padding_mask.to(device)
        B, N, _ = encodings.shape
        samples = torch.zeros(n_samples, N, dtype=torch.int).to(device)
      
        q = self.q
        ## fields shape: (B,N,q), we will consider B=1 for the moment
        ## Couplings shape: (B, N*q, N*q)
        couplings, fields = self.forward_ardca(encodings, padding_mask)
        aux1 = torch.tensor(np.arange(N), dtype=torch.float).reshape(N,1)
        aux1[0] = 1.0
        #aux2 = torch.ones(1,q, requires_grad=False)
        aux1 = torch.matmul(aux1, torch.ones(1,q))
        #aux_flat=
        aux1 = torch.matmul(aux1.reshape(N*q,1), torch.ones(1,N*q))
        aux1=torch.einsum('i,jk->ijk', torch.ones(B), aux1).to(device)
        ### AUX1 SHOULD BE [B, Nq, Nq]

        couplings = couplings/aux1
        ##############################################################################
        couplings = couplings[0, :, :] #### This simplifies
        fields = fields[0,:].reshape(N, q)
        p_pos = softmax(fields[0], dim=0)
        #p_pos = softmax(-fields[0], dim=0)

        samples[:,0] = Categorical(p_pos).sample((n_samples,))

        with torch.no_grad():   ## If you take this out you will fill you RAM linearly in n_samples
            for pos in range(1,N):
                Ham = torch.zeros(n_samples, q).to(device)
                Ham += fields[pos, :]
                for acc in range(pos):
                    ### Can we also vectorize this? Not sure it is going to help that much
                    for aa in range(q):
                        second_idx = acc*q + samples[:, acc]
                        Ham[:, aa] += couplings[pos*q+aa, second_idx]#.unsqueeze(-1)
                p_pos = softmax(Ham, dim=1)
                samples[:, pos] = Categorical(p_pos).sample()
        return samples
    

    
    
    