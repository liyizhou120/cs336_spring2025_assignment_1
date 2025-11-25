import math 
import os 
from typing import IO, BinaryIO, Callable, Iterable, Iterator, Optional
import torch 
from torch import nn 
import numpy as np 
import numpy.typing as npt 
from einops import einsum, rearrange

def cross_entropy(logits_i: torch.Tensor, target_i: torch.Tensor) -> torch.Tensor: 
    """
    Compute cross entropy for a single word in the sample 
    
    Args: 
        logits_i: [1 : x[i]] computes the unnormalized logits vector (batch_size,  ..., vocab_size)
        target_i: the correct answers indicating which one is right (batch_size, ...)
        
    Returns: 
        torch.Tensor: cross entropy tensor
    """
    # Reshape the dimensions of logits and target tensors 
    logits_i_reshaped = rearrange(logits_i, "b ... v -> (b ...) v") # (batch_size, vocab_size) 
    target_i_reshaped = rearrange(target_i, "b ... -> (b ...)") #(batch_size, )
    
    # Preprocessing logits, substracting each sample with the max logit to prevent overflowing
    logits_i_stable = logits_i_reshaped - logits_i_reshaped.max(dim=-1, keepdim = True).values 
    targets_logit = logits_i_stable.gather(1, target_i_reshaped.unsqueeze(1)).squeeze(1)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_i_stable), dim=-1))
    loss = -targets_logit + log_sum_exp
    
    return loss.mean()




class SGDOptimizer(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3): 
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr, 
        }
        super().__init__(params, defaults=defaults)
        
    def step(self, closure: Optional[Callable] = None): 
        """
        Execute one step of optimization 
        """
        
        loss = None if closure is None else closure()
        
        # Gradient descending on each parameter set 
        for group in self.param_groups: 
            lr = group["lr"]  # Parameter group has the same learning rate
            for param in group["params"]: 
                if param.grad is None: 
                    continue 
                
                state = self.state[param] #read saved state 
                t = state.get("t", 0) # read numbers of iteration 
                grad = param.grad.data # read gradient 
                param.data -= lr / math.sqrt(t + 1) * grad # Update parameters based on gradients and number of iterations, learning rates diminishing 
                state["t"] = t + 1 # iteration + 1 
                
        return loss 

def run_experiment(lr): 
    print(f"\nRunning experiment with lr = {lr}")
    weights = torch.nn.Parameter(5 * torch.rand((10,10)))
    optimizer = SGDOptimizer([weights], lr = lr)
    
    for t in range(10): 
        optimizer.zero_grad()
        loss = (weights ** 2).mean()
        loss.backward() 
        optimizer.step()
        print(f"Step {t:02d}: loss = {loss.item(): .6f}")
    

class AdamWOptimizer(torch.optim.Optimizer): 
    def __init__(self, params, lr, betas=(0.9, 0.999), weight_decay = 0.01, eps = 1e-8):
        defaults = {
            "lr": lr, 
            "betas": betas, 
            "weight_decay": weight_decay, 
            "eps": eps,
        }
        super().__init__(params, defaults)
            
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
            
        for group in self.param_groups: 
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
                
            for param in group["params"]: 
                if param.grad is None: 
                    continue
                    
                state = self.state[param]
                if len(state) == 0: 
                    #Initialization 
                    state["step"] = 1 # iteration starts 
                    state["exp_avg"] = torch.zeros_like(param.data) # first moment vector 
                    state["exp_avg_sq"] = torch.zeros_like(param.data)
                
                # Reading state and gradients 
                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = betas 
                grad = param.grad.data
                
                # Updating m and v 
                state["exp_avg"] = beta1 * exp_avg + (1 - beta1) * grad 
                state["exp_avg_sq"] = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2
                
                # Apply regularization to learning rate and compute adjusted lr
                lr_t = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                
                # Updating the parameters using lr and m and v 
                param.data -= lr_t * state["exp_avg"] / (torch.sqrt(state["exp_avg_sq"]) + eps)
                
                # Use weight decay to update the param again 
                param.data -= lr * weight_decay * param.data 
                
                # Update iteration 
                state["step"] += 1
        
        return loss 

def learning_rate_cosine_schedule(it: int,
                                  max_learning_rate: float, 
                                  min_learning_rate: float, 
                                  warmup_iters: int, 
                                  cosine_cycle_iters: int,) -> float: 
    
    """
    Cosine schedule learning rate, making dynamic adjustment of learning rate 
    
    Args: 
        it: iteration numbers 
        max_learning_rate: maximum learning rate 
        min_learning_rate: minimum learning rate 
        warmup_iters: iterations needed after warmup 
        cosine_cycle_iters: iterations needed after cosine cycle 
        
    Returns: 
        float: current learning rate 
    """
    
    if it < warmup_iters:
        # Warm-up phase, linearly increasing learning rate to max value 
        return it / warmup_iters * max_learning_rate
    
    elif it < cosine_cycle_iters: 
        # Cosine annealing, learning rate smoothly moving from max to min 
        cos_percent = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * ( 
                1 + math.cos(math.pi * cos_percent)                                                                    )
        
    else: 
        # keep min learning rate after cosine period 
        return min_learning_rate
         
def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float = 1.0, eps: float = 1e-6) -> float: 
    """
    Gradient clipping to prevent explosion 

    Args:
        params (Iterable[torch.nn.Parameter]): parameter list 
        max_norm (float, optional): maximum gradient norm 
        eps (float, optional): _description_. Defaults to 1e-6.

    Returns:
        float: _description_
    """
    total_norm = 0.0 
    # Compute L2 Norm of the parameter gradient 
    for param in params: 
        if param.grad is not None: 
            total_norm += torch.sum(param.grad ** 2)
            
    total_norm = total_norm ** 0.5 
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0: 
        for param in params: 
            if param.grad is not None: 
                param.grad.data.mul_(clip_coef)
    

def get_batch(dataset:npt.NDArray, batch_size: int, context_length:int, device: str) -> tuple[torch.Tensor, torch.Tensor]: 
    """
    Taken a token sequence, sampling batches of tokens with context length by using sliding window
     

    Args:
        dataset (npt.NDArray): input token sequence
        batch_size (int): batch size
        context_length (int): context length
        device (str): device GPU

    Returns:
        tuple[torch.Tensor, torch.Tensor]: return meta data, with 2 tensors: 
            - Input sequence tensor, with shape (batch_size, context_length)
            - Output sequence tensor, with shape(batch_size, context_length),targeting the next token in the sequence 
    """
    dataset_len = dataset.shape[0]
    if dataset_len < context_length: 
        raise ValueError(f"Dataset length {dataset_len} is less than context length {context_length}")
    
    starts = np.random.randint(0, dataset_len - context_length, size = batch_size)
    inputs = np.stack( [dataset[start:start + context_length] for start in starts], dtype = np.int64)
    targets = np.stack([dataset[start + 1:start + context_length + 1] for start in starts], dtype = np.int64)

    return(
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device)
    )
    
def save_checkpoint(
    model:torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | BinaryIO | IO[bytes]
): 
    """
    save model weights, optimizer state, and iteration numbers in a dictionary, to a didicated folder or object 
    
    """
    return torch.save({
        "model_state_dict": model.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(), 
        "iteration": iteration, 
    }, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
)-> int: 
    """
    Load model weights, optimizer state, and iteration numbers in a dictionary, to a didicated folder or object 
    """
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None: 
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

def evaluate_model(model: nn.Module, dataset, device, batch_size, context_length, num_batches = 10): 
    model.eval() # evaluation mode 
    total_loss = 0.0 
    with torch.no_grad(): 
        for _ in range(num_batches):
            # get data from validation set 
            inputs, targets = get_batch(
                dataset, 
                batch_size = batch_size,
                context_length = context_length, 
                device = device
            )
            # Forward pass 
            logits = model(inputs)
            # Compute losses 
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()

    model.train()
    return total_loss / num_batches

def compute_grad_norm(parameters: Iterable[nn.Parameter]): 
    """
    Compute L2 Norm 
    """
    total_norm = 0.0 
    for p in parameters: 
        if p.grad is not None: 
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5 
        return total_norm 
    
                
if __name__ == "__main__":
    # for lr in [1e0, 1e1, 1e2, 1e3]:
    #     run_experiment(lr)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = SGDOptimizer([weights], lr = 1e1)
    opt = AdamWOptimizer([weights], lr = 1e1, weight_decay=0.01)
    for t in range(100): 
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(f"t={t}, loss={loss.cpu().item}")
        loss.backward()
        opt.step()
