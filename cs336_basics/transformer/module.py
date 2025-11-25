import torch 
import torch.nn as nn 
import jaxtyping
from math import sqrt 
from einops import einsum, reduce, rearrange 


class Linear(nn.Module):
    """
    不包含bias的线性变换层，y=Wx=xW^T
    """ 
    def __init__(self, in_features : int, out_features: int, device= None, dtype= None): 
        """
        Initialization 
        Args:
            in_features (int): Input feature 
            out_features (int): Output feature 
            device (None): CPU/GPU Device 
            dtype (None): Data Type 
        """
        super().__init__() 
        self.d_in = in_features
        self.d_out = out_features
        self.W = nn.Parameter(torch.empty((out_features, in_features), device = device, dtype = dtype)) # 内存中in维度是连续的，有助于加速矩阵乘计算
      
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Forward Pass 
        Arg: 
        x: input Tensor, shape(batch, ...., in_features)
        
        """
        
        return einsum(x, self.W, 'batch ... input, output input -> batch ... output')

class Embedding(nn.Module): 
    """
    Embedding layer 

    Args:
        num_embeddings: embedding vocab size 
        embedding_dim: embedding vector dimensions (vector size)
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        #A single row in the table stands for embedding vector for a word
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = None)) 
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Forward Pass 

        Args:
            x (torch.Tensor): Input Tensor,  shape(batch, seq_len), indexing the vocab table 

        Returns:
            torch.Tensor
        """
        batch_size = x.shape[0]
        return torch.stack([torch.index_select(self.weight, dim = 0, index=x[i]) for i in range(batch_size)])

class RMSNorm(nn.Module): 
    """
    Root-Mean-Square Normalization 

    Args:
        
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None): 
        """
        Initialize RMSNorm 

        Args:
            d_model (int): Dimensions of input features 
            eps (float, optional): a small number to stay away from zeroing
        """
        
        super().__init__()
        self.d_model = d_model 
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Forward Pass 

        Args:
            x (torch.Tensor): Input tensor, with shape(batch, seq_len, d_model)

        Returns:
            torch.Tensor: _description_
        """
        
        in_dtype = x.dtype
        x = x.to(dtype = torch.float32)
        # Compute RMS 
        rms = torch.sqrt(reduce(x ** 2, "b ... d -> b ... 1", 'mean') + self.eps)
        # Normalize
        x = x / rms * self.weight
        
        return x.to(in_dtype) # convert dtype back 

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor: 
   """
   Compute softmax 
   Args: 
       x: input tensor 
       dim: Compute dimensions of softmax 
   Returns: 
       tensors after softmax layer 
   """ 

   # Subtract max value to prevent overflow (adding constant c not changing softmax output)
   x_max = torch.max(x, dim=dim, keepdim=True).values 
   x_exp = torch.exp(x - x_max)
   return x_exp/torch.sum(x_exp, dim=dim, keepdim=True)
   
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask):
    """
    Compute scaled dot product attention 

    Args: 
        Q: Query tensors, with shape (batch, ..., seq_len_q, d_k)
        K: Key tensors, with shape(batch, ..., seq_len_k, d_k)
        V: Value tensors, with shape(batch,..., seq_len_k, d_v)
        mask: mask tensors, with shape (seq_len_q, seq_len_k)
    Returns: 
        Attention Output tensors, with shape (batch, ..., seq_len_q, d_v)
    """
    scores = einsum(Q, K, 'batch ... q d_k, batch ... k d_k -> batch ... q k')
    d_k = Q.shape[-1]
    scores /= sqrt(d_k) #Scale QK^T 
    
    #Applying mask: 
    if mask is not None: 
        scores = scores.masked_fill(mask==0, float('-inf'))
    
    attn_weights = softmax(scores, dim=-1)
    output = einsum(attn_weights, V, 'batch ... q k, batch ... k d_v -> batch ... q d_v')
    
    return output 

class MultiHeadSelfAttention(nn.Module): 
    """
    Multihead Self Attention 
    """
    
    def __init__(self, d_model: int, num_heads: int, device = None, dtype = None): 
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads #d_v = d_k
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.device = device 
        self.dtype = dtype
        
        self.w_q = Linear(d_model, self.num_heads * self.d_k, device = device, dtype = dtype)
        self.w_k = Linear(d_model, self.num_heads * self.d_k, device = device, dtype = dtype)
        self.w_v = Linear(d_model, self.num_heads * self.d_k, device = device, dtype = dtype) 
        self.w_qkv = Linear(d_model, self.num_heads * self.d_k * 3, device = device, dtype = dtype) 
        self.w_o = Linear(self.num_heads * self.d_k, self.d_model, device=device, dtype = dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: shape(batch, ..., seq_len, d_model) 
        """
        seq_len = x.shape[-2]
        QKV = self.w_qkv(x)
        # Segment Q,K,V 
        Q, K, V = rearrange(QKV, "... seq_len(three head d_k) -> three ... head seq_len d_k", three = 3, head = self.num_heads)
        
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).to(self.device)
        
        atten = scaled_dot_product_attention(Q, K, V, mask)
        
        #Put multi-head attention back 
        atten = rearrange(atten, "... head seq_len d_k -> ... seq_len(head d_k)")
        
        return self.w_o(atten) 
            



class RoPE(nn.Module): 
    """
    Rotatary positional embedding  

    Args:
       Theta: rotation angles 
       d_k: dimensions of Q or K matrix 
       max_seq_len: maximum sequence length 
    """
    def __init__(self, theta:float, d_k: int, max_seq_len:int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k 
        self.max_seq_len = max_seq_len
        self.device = device 
        
        # Compute Rotary Positional Matrix and put it into buffer 
        self.register_buffer("rope", self._precompute_freqs_cis(), persistent=False)
    
    def _precompute_freqs_cis(self) -> torch.Tensor: 
        """
        预计算频率和相位
        Returns:
            形状为(max_seq_len, d_k)的张量，包含旋转位置编码
        """
        # 计算\theta_i序列，也就是频率序列
        # theta_i = 1 / { theta^{2i / d_k} }
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device)[:(self.d_k // 2)] / self.d_k))
        # 生成序列索引m [0, 1, ..., max_seq_len-1]
        seq_idx = torch.arange(0, self.max_seq_len, device=self.device)
        # 计算 m * \theta_i 矩阵
        freqs = einsum(seq_idx, freqs, "seq, d -> seq d")

        # 复数化
        # freqs[m][i] = m * \theta_i
        # freqs_cis[m][i] = 1 * e^{i * m * \theta_i}
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        """
        Forward pass 

        Args:
            x (torch.Tensor): Input tensor, shape(..., seq_len, d_k)
            token_positions (torch.Tensor): position index, shape(..., seq_len)

        Returns:
            RoPE tensor, shape(..., seq_len, d_k)
        """
        #Group different dimensions 
        x_ = rearrange(x, "... seq (d two) -> ... seq d two", two = 2).float()
        x_ = torch.view_as_complex(x_)
        rope_pos = self.rope[token_positions] # (batch, ..., seq_len, d_k // 2)
        x_out = rearrange(torch.view_as_real(x_ * rope_pos), "... seq d two -> ... seq (d two)", two = 2)

        return x_out.to(x.dtype)
        
        
class SWiGLUFeedForward(nn.Module): 
    """
    Feed Forward using SWiGLU activations 
    """ 
    def __init__(self, d_model: int, d_ff: int = None, device = None, dtype = None): 
        """
        Initialization

        Args:
            d_model (int): Input feature dimensions
            d_ff (int): SWiGLU inner feedforward layer, if none then equal to 8/3 * d_model (approximate to 64)
            
        """
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(8 / 3 * d_model)
            self.d_ff = (self.d_ff + 63) // 64 * 64
        else:
            self.d_ff = d_ff
        self.weight1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.weight2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.weight3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """

        Forward pass W2(SiLU(W1 x) \odot (W3 x))
        Args:
            x (torch.Tensor): with shape(batch, ... , d_model)

        Returns:
            torch.Tensor: with shape (batch, ..., d_model)
        """
        w1_x = self.weight1(x)
        w3_x = self.weight3(x)
        silu = w1_x * torch.sigmoid(w1_x)
        swiglu = silu * w3_x 
        return self.weight2(swiglu)
    
class MultiHeadAttentionwithRoPE(MultiHeadSelfAttention):
    
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len:int, device = None, dtype = None): 
        super().__init__(d_model, num_heads, device = device, dtype = dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        """
        Forward Pass 

        Args:
            x (torch.Tensor): input embedding with shape(batch, ... , seq_len, d_model)
            token_positions (torch.Tensor): index reference with shape (batch, ..., seq_len)

        Returns:
            Input embedding, shape(batch, ... , seq_len, d_model)
        """
        
        seq_len = x.shape[-2]
        QKV = self.w_qkv(x)
        Q, K, V = rearrange(QKV, "... seq_len (three head d_k) -> three ... head seq_len d_k", three = 3, head = self.num_heads)
        
        # Use RoPE on Q, K, with head being the dimension of batch 
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        #masking the QK matrix
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype = torch.bool)).to(self.device)
        
        atten = scaled_dot_product_attention(Q, K, V, mask)
        
        #put heads back 
        atten = rearrange(atten, "... head seq_len d_k -> ... seq_len (head d_k)")
        
        return self.w_o(atten)
    
    
class TransformerBlock(nn.Module):
        """
        Transformer block, with self attention and feedforward network

        """
        def __init__(self, d_model: int, num_heads: int, max_seq_len: int, d_ff: int = None, theta: float = 10000.0, device = None): 
            
            """
            Initialization of Transformer block 
            Args: 
                d_model: input embeddings 
                num_heads: number of heads 
                d_ff: FFN dimensions 
                max_seq_len: maximum sequence length 
                theta: rotation angles 
            
            """
            
            super().__init__()
            self.d_model = d_model 
            self.num_heads = num_heads 
            self.d_ff = d_ff 
            
            self.attention_rope = MultiHeadAttentionwithRoPE(d_model, num_heads, theta, max_seq_len, device = device)
            self.ffn = SWiGLUFeedForward(d_model, d_ff, device = device)
            self.norm1 = RMSNorm(d_model, device=device)
            self.norm2 = RMSNorm(d_model, device=device)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor: 
            """
            FFN 
            Args:
                x: input tensors, (batch, ..., seq_len, d_model)
                token_positions: position index 

            Returns:
                torch.Tensor: _description_
            """
            token_positions = torch.arange(x.shape[-2], dtype = torch.int, device = x.device) #(batch, ..., seq_len)
            
            # Multihead Attention 
            attn_output = self.attention_rope(
                self.norm1(x), token_positions
            )
            x2 = x + attn_output 
            ffn_output = self.ffn(self.norm2(x2))
            return x2 + ffn_output


class TransformerLM(nn.Module): 
    """
    Transformer model 
    """    
    def __init__(self, 
                 vocab_size: int, 
                 context_length: int, 
                 num_layers: int, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int = None, 
                 rope_theta: float = 10000.0,
                 device = None, 
                 dtype = None): 
        """
        Initializing Transformer model

        Args:
            vocab_size (int): vocab size 
            context_length (int): max context length 
            num_layers (int): number of layers of Transformer
            d_model (int): dimensions of input features
            num_heads (int): number of heads 
            d_ff (int, optional): FFN dimensions 
            rope_theta (float, optional): rope angles 
            device (_type_, optional): device: CPU or GPU 
            dtype (_type_, optional): data type 
        """
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model, device = device)
        self.tf_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, d_ff, rope_theta, device=device)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device)
        self.out_embedding = Linear(d_model, vocab_size, device=device)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Forward pass 

        Args:
            x (torch.Tensor): input dimensions, with shape (batch, seq_len) and each element being index in the vocab table
        
        Returns:
            torch.Tensor: Output embeddings, with shape(batch, seq_len, vocab_size)
            
        """
        # Embedding 
        x = self.token_embedding(x)
        # Transformer blocks 
        for block in self.tf_blocks: 
            x = block(x)
        # Linearization 
        x = self.ln_final(x)
        # Output layer
        x = self.out_embedding(x)
        # softmax      
        # return softmax(x, dim= -1)
        return x 
    