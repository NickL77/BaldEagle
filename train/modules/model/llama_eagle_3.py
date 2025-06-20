import torch
import torch.nn as nn

from functools import partial
from typing import List, Optional, Union, Tuple

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig, LlamaDecoderLayer, LlamaRMSNorm, LlamaAttention, LlamaMLP

from transformers.cache_utils import Cache

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # change qkv_proj to take in hidden_size * 2 instead of hidden_size
        output_dim_query = self.self_attn.q_proj.out_features
        output_dim_key_value = self.self_attn.k_proj.out_features
        self.self_attn.q_proj = nn.Linear(config.hidden_size * 2, output_dim_query, bias=config.attention_bias)
        self.self_attn.k_proj = nn.Linear(config.hidden_size * 2, output_dim_key_value, bias=config.attention_bias)
        self.self_attn.v_proj = nn.Linear(config.hidden_size * 2, output_dim_key_value, bias=config.attention_bias)
        
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # add embeds for token embeddings
    # embeds is the original hidden_states and hidden_states is the fused features hidden_states
    def forward(
        self,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states)
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        # Remove post_init as it's weight initialization makes model worse
        # Change to use mid_layer instead of layers (force to 1 layer)
        # Change qkv_proj to take in hidden_size * 2 instead of hidden_size

        # self.init_weights = lambda: print("Skipping init weights")
        super().__init__(config)
        # these should all be defined in the super class
        # self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # self.gradient_checkpointing = False

        del self.layers
        self.midlayer = LlamaDecoderLayer(config, 0)

        self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        eagle_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs
    ) -> BaseModelOutputWithPast:
        # hidden_states is the embeded token embeddings
        # eagle_hidden_states is the eagle fused hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            raise ValueError("no support for caching")
            # past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # eagle_hidden_states should already be concatenated (from dataloader)
        if eagle_hidden_states.shape[-1] != hidden_states.shape[-1]:
            eagle_hidden_states = self.fc(eagle_hidden_states)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                partial(self.midlayer.__call__, **flash_attn_kwargs),
                hidden_states,
                eagle_hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = self.midlayer(
                hidden_states,
                eagle_hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LlamaForCausalLMEagle3(PreTrainedModel):
    def __init__(self, config):
        """
        Initializes the model with the Hugging Face structure:
        
          LlamaForCausalLM(
            (model): LlamaModel(...)
          )
        """
        super().__init__(config)
        self.gradient_checkpointing = True
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        config.attn_implementation="flash_attention_2"

        # Monkey patch post init to no op - Hugging Face weight initialization makes model worse
        # def noop_post_init(self):
        #     print("Running no op post init")
        #     pass
        # LlamaModel.post_init = noop_post_init

        self.model = LlamaModel(config)
        
        # Eagle 3 now has the norm layer in model and input_layernorm in layers/mid_layer
        # del self.model.norm
        # setattr(self.model, "norm", lambda x: x)
        # del self.model.layers[0].input_layernorm
        # setattr(self.model.layers[0], "input_layernorm", lambda x: x)
    
    def load_embedding_weights(self, weights):
        self.model.embed_tokens.weight = nn.Parameter(weights)

    def forward(
            self,
            hidden_state: torch.Tensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if use_cache is not True:
        #     print("Warning: use_cache is not True, setting to True")
        #     use_cache = True

        # token_emb = self.model.embed_tokens(input_ids)
        # concat = torch.cat([token_emb, hidden_state], dim=-1)
        
        # proj = self.model.fc(concat)

        outputs = self.model(
            input_ids=input_ids,
            eagle_hidden_states=hidden_state,
            # inputs_embeds=proj,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs, 
        )        # return outputs

        # return outputs

        hidden_states = outputs[0]
        return hidden_states
