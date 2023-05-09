import torch
from models import T5VQGSPT, VAE_CONFIG
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-base')

vae_config=VAE_CONFIG(initialize_from_vocab=True, n_soft_prompts=50)
model = T5VQGSPT.from_pretrained(
        pretrained_model_name_or_path='t5vqgspt/BM25-0_P-50_Z-128_BS-4-FREEZE/checkpoint-1000',
        vae_config=vae_config,
        tokenizer=tokenizer, 
).eval()

print(model.encoder.embed_tokens.soft_prompt_embeds[0, -10:])
print(model.decoder.embed_tokens.weight[0, -10:])
print(model.lm_head.weight)

with open('param.txt', 'w') as f:
    for name, param in model.named_parameters():
        f.write(name+'\n')
# print(model.encoder.embed_tokens.hidden2mean.weight)

text = 'The huge carcass has become a macabre tourist attraction with families approaching the animal to look and people taking selfies. There were concerns the marine mammal would have to be dissected to remove it from the beach, but East Riding of Yorkshire Council has said it hopes to move it whole.'

model = T5VQGSPT.from_pretrained(
        pretrained_model_name_or_path='t5vqgspt/BM25-0_P-50_Z-128_BS-4-FREEZE/checkpoint-2000',
        vae_config=vae_config,
        tokenizer=tokenizer, 
).eval()

print(model.encoder.embed_tokens.soft_prompt_embeds[0, -10:])
print(model.decoder.embed_tokens.weight[0, -10:])
print(model.lm_head.weight)

# print(model.encoder.embed_tokens.soft_prompt_embeds[0, -10:])
# # o=model.generate(**tokenizer(text, return_tensors='pt'), do_sample=True, top_k=10)
# o=model.generate(**tokenizer(text, return_tensors='pt'), num_beams=2)
# print("\n".join([tokenizer.decode(o[i], skip_special_tokens=True) for i in range(len(o))]))
#
# model.encoder.embed_tokens.soft_prompt_embeds = torch.nn.Parameter(
#         torch.ones(model.encoder.embed_tokens.soft_prompt_embeds.shape)
# )
# print(model.encoder.embed_tokens.soft_prompt_embeds[0, -10:])
# o=model.generate(**tokenizer(text, return_tensors='pt'), num_beams=2)
# print("\n".join([tokenizer.decode(o[i], skip_special_tokens=True) for i in range(len(o))]))
