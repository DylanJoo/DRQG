from models import BartVQGSPT, VAE_CONFIG
from transformers import BartTokenizer, BartConfig

config = BartConfig.from_pretrained('facebook/bart-base')
model = BartVQGSPT.from_pretrained('bartvqgspt/test/checkpoint-5000', config=config, vae_config=VAE_CONFIG)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

passage = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."

input = tokenizer(passage, return_tensors='pt')
output = model.generate(input.input_ids, attention_mask=input.attention_mask)
for i in range(len(output)):
    print(tokenizer.decode(output[i]))
