import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, Gemma2ForCausalLM

model_type = 'gemma'
model_name = "google/gemma-2-2b"
model_type = Gemma2ForCausalLM

shared_kwargs = dict(
    cache_dir=os.path.expanduser("~/.cache/"),
    token="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",
    trust_remote_code=True,
)

print(AutoConfig.from_pretrained(model_name, **shared_kwargs))

# make config smaller
print('--' * 20)

for latent_type in ['concat_seq']:
    print(f"Latent type: {latent_type}")
    config = AutoConfig.from_pretrained(model_name, **shared_kwargs, attn_implementation='eager')
    
    assert hasattr(config, 'num_hidden_layers') 
    assert hasattr(config, 'latent_type')
    # assert hasattr(config, 'latent_layer_ratio')
    
    config.num_hidden_layers = 1
    config.latent_type = latent_type
    config.latent_layer_ratio = 0.0

    # load model
    model = model_type(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **shared_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id

    emb_size = model.config.hidden_size
    print(f"Embedding size: {emb_size}")

    text_to_encode = ["Hello, my dog is cute", "I like to go for a walk"]
    inputs = tokenizer(text_to_encode, return_tensors="pt", padding=True, truncation=True, max_length=128)

    latents = torch.randn(2, emb_size)
    logits = model(**inputs, latents=latents).logits

    print(logits.shape)
    print('--' * 20)
    del model, tokenizer, config, logits, latents, inputs