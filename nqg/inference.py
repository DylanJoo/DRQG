from tqdm import tqdm
from datacollator import DataCollatorForT5VQG
from models import T5VQG
from transformers import AutoConfig, AutoTokenizer
from dataclasses import dataclass, field
from datasets import Dataset

PATH="t5vqg_v0/checkpoint-10000"
device='cpu'

vae_config = {"latent_size" : 128, 
              "k": 0.0025,
              "x0": 2500,
              "annealing_fn": 'logistic'}
@dataclass
class OurModelArguments:
    latent_size: int = field(default=256)
    k: float = field(default=0.0025)
    x0: int = field(default=2500)
    annealing_fn: str = field(default='logistic')

config = AutoConfig.from_pretrained('t5-base')
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = T5VQG.from_pretrained(
        pretrained_model_name_or_path=PATH,
        config=config,
        vae_config=OurModelArguments,
        tokenizer=tokenizer
)

text = ["<PAD>", "It is generally safe for pregnant women to eat chocolate because studies have shown to prove certain benefits of eating chocolate during pregnancy. However, pregnant women should ensure their caffeine intake is below 200 mg per day."]

mylist = [{"passage": f"<extra_id_10> {p}"} for p in text]
dataset = Dataset.from_list(mylist)

from datacollator import DataCollatorForT5VQG
data_collator = DataCollatorForT5VQG(
        tokenizer=tokenizer, 
        padding=True,
        return_tensors='pt',
        is_eval=True,
        is_test=True
)

from torch.utils.data import DataLoader
dataloader = DataLoader(
        dataset,
        batch_size=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=data_collator
)

for batch in tqdm(dataloader):
    passages_info = batch.pop('passage')
    positive_info = batch.pop('positive')
    negative_info = batch.pop('negative')

    for k in batch:
        batch[k] = battch[k].cuda(device)

    h = model.encoder(**batch)
    o = model.generate(encoder_output=h)

    for i, tokens in enumerate(o):
        print(passage_info[i])
        print(positive_info[i])
        print(negative_info[i])
        print(tokenizer.decode(o[i], skip_special_tokens=True))

