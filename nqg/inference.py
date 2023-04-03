from tqdm import tqdm
from datacollator import DataCollatorForT5VQG
from models import T5VQG
from transformers import AutoConfig, AutoTokenizer
from dataclasses import dataclass, field
from datasets import Dataset

PATH="t5vqg_v0/checkpoint-10000"
PATH='t5-base'
device='cuda'
output_positive=True

vae_config = {"latent_size" : 256, 
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
).to(device)
model.eval()

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

    for k in batch:
        batch[k] = batch[k].to(device)

    oh = model.encoder(**batch)

    hidden_states = oh[0]
    # add gaussian
    if output_positive:
        mean = model.hidden2pmean(hidden_states)
        logv = model.hidden2plogv(hidden_states)
        std = torch.exp(0.5 * logv)

    else:
        mean = model.hidden2nmean(hidden_states)
        logv = model.hidden2nlogv(hidden_states)
        std = torch.exp(0.5 * logv)

    # decode
    z = torch.cat(((std+mean), mean, (-std+mean)), 0)
    bs, sl, d_model = z.shape
    h = self.latent2hidden(z) 
    zeros = torch.zeros(bs, sl, d_model).to(hidden_states.device)
    residuals = torch.cat((h, zeros), 1)
    encoder_outputs.last_hidden_state = \
            residuals + hidden_states[0].repeat((3, 1, 1))

    o = model.generate(encoder_outputs=oh)

    for i, passage in enumerate(passages_info):
        print("Passages\t", passages)
        print("Predicted query1\t", tokenizer.decode(o[i*3+0], skip_special_tokens=True))
        print("Predicted query2\t", tokenizer.decode(o[i*3+1], skip_special_tokens=True))
        print("Predicted query3\t", tokenizer.decode(o[i*3+2], skip_special_tokens=True))


