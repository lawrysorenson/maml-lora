
from transformers import BartModel, BartConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from lora import initialize_lora

config = config = BartConfig(
    vocab_size = 10, # must be bigger than token_ids
    max_position_embeddings = 512,
    encoder_layers = 6,
    encoder_ffn_dim = 2048,
    encoder_attention_heads = 8,
    decoder_layers = 6,
    decoder_ffn_dim = 2048,
    decoder_attention_heads = 8,
    encoder_layerdrop = 0.0,
    decoder_layerdrop = 0.0,
    activation_function = 'swish',
    d_model = 512,
    dropout = 0.1,
    attention_dropout = 0.0,
    activation_dropout = 0.0,
    init_std = 0.02,
    classifier_dropout = 0.0,
    scale_embedding = True,
    pad_token_id = 1,
    bos_token_id = 1,
    eos_token_id = 2,
    is_encoder_decoder = True,
    decoder_start_token_id = 1,
    forced_eos_token_id = 2
)

# seperate embeddings

# only supports one adapter per batch
class AdapterModel(nn.Module):
    def __init__(self, eng_dim=16000, o_dim=8000, d_model=512):
        super().__init__()

        self.lang_mapper = {'eng': 0, 'oth': 1}

        self.scale_embedding = d_model**0.5

        self.eng_embed = nn.Parameter(torch.zeros((eng_dim, d_model)))
        nn.init.normal_(self.eng_embed)
        self.eng_lm_head = nn.Linear(d_model, eng_dim)

        self.tgt_embed = nn.Parameter(torch.zeros((o_dim, d_model)))
        nn.init.normal_(self.tgt_embed)

        self.model = BartModel(config)

        # other language in encoder
        # store as a normal list
        self.lora_mods = initialize_lora(self.model.encoder)
        # self.lora_mods = initialize_lora(self.model.decoder)

    def inner_params(self):
        ans = [self.tgt_embed]
        for l in self.lora_mods: ans += l.lora_params()
        return ans


    def outer_params(self):
        exc = set(self.inner_params())
        ans = [p for p in self.parameters() if p not in exc]
        return ans
    
    def reset_adapter(self):
        nn.init.normal_(self.tgt_embed)
        for l in self.lora_mods: l.reinit()

    def forward(self, input_ids, input_mask, dec_input_ids, dec_input_mask, input_langs, output_langs):

        for l in input_langs: assert l == 'oth'
        input_embeds = F.embedding(input_ids, self.tgt_embed) * self.scale_embedding

        for l in output_langs: assert l == 'eng'
        dec_input_embeds = F.embedding(dec_input_ids, self.eng_embed) * self.scale_embedding


        model_output = self.model(
            inputs_embeds=input_embeds,
            attention_mask=input_mask,
            decoder_inputs_embeds=dec_input_embeds,
            decoder_attention_mask=dec_input_mask
        )

        out_embeds = model_output.last_hidden_state

        for l in output_langs: assert l == 'eng'
        logits = self.eng_lm_head(out_embeds)
        
        return logits


    def generate(self, input_ids, input_mask):
        pass


