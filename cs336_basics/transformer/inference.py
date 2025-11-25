import torch
from loguru import logger
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer.module import TransformerLM, softmax
from cs336_basics.transformer.trainer_util import load_checkpoint


def nucleus_sampling(probs, top_p):
    """
    Implementation of Nucleus sampling (Top-p sampling)
    Rank top-p token probabilities from top to bottom, adding every single one until the sum surpassing top-p.
    All token probabilities are set to zero after that

    Args:
        probs (_type_): probability distribution of shape (batch_size, vocab_size)
        top_p (_type_): _description_
    
    Return: 
        Probability distribution after top-p 
    """
    
    #Sorting each token of every sample (descending order)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative probability
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices with cumulative sum that is less than top-p, marking it as true
    nucleus_mask = cumulative_probs <= top_p
    
    # if the current index has cumulative probability > top-p (ex: 0.95 > top_p = 0.9), then probabilities after this are all set to zero
    # Save the token with the highest probability just in case 
    nucleus_mask[:, 0] = True 
    
    # Set all indice after to 0 
    sorted_probs_fixed = sorted_probs * nucleus_mask
    
    # normalized probability distribution 
    sorted_probs_sum = torch.sum(sorted_probs_fixed, dim = -1, keepdim = True)
    sorted_probs_normalized = sorted_probs_fixed / sorted_probs_sum
    
    probs_filtered = torch.zeros_like(probs)
    probs_filtered.scatter_(1, sorted_indices, sorted_probs_normalized)
    
    return probs_filtered

if __name__ == "__main__":
    
    prompts = ["The quick brown fox jumps over the lazy dog",
               "Once upon a time,",
               "Tom and Lily are best friends.",]
    
    max_new_tokens = 128 
    temperature = 1.2 
    top_p = 0.9 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context_length = 256 
    
    end_token = "<|endoftext|>"
    
    # Initializing tokenizer 
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/output/TinyStories_train_10000_vocab.bin",
        mergers_filepath= "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/output/TinyStories_train_10000_token_merges.bin",
        special_tokens=["<|endoftext|>"]
    )
    
    # Initializing model 
    model = TransformerLM(
        vocab_size=10000, 
        context_length=context_length, 
        num_layers=4, 
        num_heads=16, 
        d_model=512, 
        d_ff = 1344, 
        rope_theta = 10000, 
        device = device
    )
    
        # 加载模型参数
    load_checkpoint(
        src="/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/model/final_model_v0.pt",
        model=model,
        optimizer=None  
    )
    
    model.to(device)
    # Setting model to eval mode 
    model.eval()
    
    #Encoding the input 
    
    
    
    
    