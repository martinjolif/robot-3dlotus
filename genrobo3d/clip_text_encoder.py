import torch
import numpy as np
import open_clip
import os
from transformers import CLIPModel, AutoTokenizer, CLIPProcessor
#Encode the text instruction for 3D-Lotus tasks

model_name='ViT-B-32'
pretrained='openai'

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
model,_, preprocess =  open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
tokenizer = open_clip.get_tokenizer(model_name)

model.to(device)
model.eval()

#give your instruction here
text = tokenizer(["reach the green target", "reach the green thing"], context_length=77).to(device)


cast_dtype = model.transformer.get_cast_dtype()
x = model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
x = x + model.positional_embedding.to(cast_dtype)
x = model.transformer(x, attn_mask=model.attn_mask)
x = model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
num_tokens = text.argmax(dim=-1) + 1  # (eot_token is the highest number in each sequence)
x = [v[:num_tokens[i]] for i, v in enumerate(x)]

#with torch.no_grad():
#    text_features = model.encode_text(text)

print(len(x))
print(x[0].shape)
dict = {
        "reach the green target": x[0].detach().cpu().numpy(),
        "reach the green things": x[1].detach().cpu().numpy()
        }

np.save("embeddings.npy", dict)
