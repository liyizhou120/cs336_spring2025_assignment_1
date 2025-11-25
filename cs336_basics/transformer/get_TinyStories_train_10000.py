from cs336_basics.tokenizer import BPETokenizer 



if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab_path = "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/output/TinyStories_train_10000_token_vocab.bin"
    mergers_path = "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/output/TinyStories_train_10000_merges.bin"
    tokenizer = BPETokenizer.from_files(vocab_path, mergers_path, special_tokens)

    # Generate training set
    input_path = "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/TinyStoriesV2-GPT4-train.txt"
    output_path = "/media/yizhouli/1TB 970 Evo Plus/code/cs336/data/token/TinyStories_train_10000_token_ids.npy"
    tokenizer.encode_to_npfile(input_path, output_path)
    
    