import torch
from perceiver_pytorch import PerceiverLM

model = PerceiverLM(
    num_tokens = 20000,          # number of tokens
    dim = 32,                    # dimension of sequence to be encoded
    depth = 6,                   # depth of net
    max_seq_len = 2048,          # maximum sequence length
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,         # number of dimensions per cross attention head
    latent_dim_head = 64,        # number of dimensions per latent self attention head
    weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
)

class VocabReducer:
    def __init__(self, input_vocab_size):
        self.map = torch.zeros(input_vocab_size, dtype=torch.long)
        self.count = 0
    def __call__(self, inputs):
        missing = (self.map[inputs] == 0) # returns boolean (vocab_size)
        if torch.any(missing):
            missing = torch.unique(inputs[missing])
            newcount = self.count + len(missing)
            self.map[missing] = torch.arange(self.count, newcount) + 1
            self.count = newcount
            print('Vocab size:', newcount)
        return self.map[inputs].view(*inputs.shape) - 1

#seq = torch.randint(0, 20000, (1, 512))
#mask = torch.ones(1, 512).bool()

import datasets
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.mask_token = tokenizer.eos_token
vocab_reducer = VocabReducer(tokenizer.vocab_size)
mask_token = vocab_reducer(torch.tensor([tokenizer.mask_token_id]))[0]
model.train()
optim = torch.optim.SGD(model.parameters(), lr=0.0001)
batchsize = 16
batch = []
for line in (item['text'] for item in datasets.load_dataset('wikitext', 'wikitext-103-raw-v1')['train']):
    batch.append(line)
    if len(batch) < 16:
        continue
    token_ids, attention_mask = (value for name, value in tokenizer(batch, padding=True, return_tensors='pt').items())
    token_ids = vocab_reducer(token_ids)
    # mask 15%
    masked_idcs = torch.logical_and(torch.rand(token_ids.shape) <= 0.15, attention_mask != 0)
    label_ids = token_ids[masked_idcs].clone()
    token_ids[masked_idcs] = mask_token
    attention_mask[masked_idcs] = 0
    
    logits = model(token_ids, mask = attention_mask) # (1, 512, 20000)

    
