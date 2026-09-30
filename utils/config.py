def get_config():
    return {
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "vocab_size": 10000,
        "max_seq_len": 512,
        "seq_len": 512,
        "d_ff": 2048,
        "rope_base": 10000,
    }