# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import Attention
from diffusers.models.attention import FeedForward, AdaLayerNorm, AdaLayerNormZero

from einops import rearrange, repeat


@dataclass
class Transformer3DModelOutput(BaseOutput):
	sample: torch.FloatTensor


if is_xformers_available():
	import xformers
	import xformers.ops
else:
	xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
	@register_to_config
	def __init__(
			self,
			num_attention_heads: int = 16,
			attention_head_dim: int = 88,
			in_channels: Optional[int] = None,
			num_layers: int = 1,
			dropout: float = 0.0,
			norm_num_groups: int = 32,
			cross_attention_dim: Optional[int] = None,
			attention_bias: bool = False,
			activation_fn: str = "geglu",
			num_embeds_ada_norm: Optional[int] = None,
			use_linear_projection: bool = False,
			only_cross_attention: bool = False,
			upcast_attention: bool = False,
	):
		super().__init__()
		self.use_linear_projection = use_linear_projection
		self.num_attention_heads = num_attention_heads
		self.attention_head_dim = attention_head_dim
		inner_dim = num_attention_heads * attention_head_dim
		
		# Define input layers
		self.in_channels = in_channels
		
		self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
		if use_linear_projection:
			self.proj_in = nn.Linear(in_channels, inner_dim)
		else:
			self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
		
		# Define transformers blocks
		self.transformer_blocks = nn.ModuleList(
			[
				BasicTransformerBlock(
					inner_dim,
					num_attention_heads,
					attention_head_dim,
					dropout=dropout,
					cross_attention_dim=cross_attention_dim,
					activation_fn=activation_fn,
					num_embeds_ada_norm=num_embeds_ada_norm,
					attention_bias=attention_bias,
					only_cross_attention=only_cross_attention,
					upcast_attention=upcast_attention,
				)
				for d in range(num_layers)
			]
		)
		
		# 4. Define output layers
		if use_linear_projection:
			self.proj_out = nn.Linear(in_channels, inner_dim)
		else:
			self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
	
	def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
		# Input
		assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
		video_length = hidden_states.shape[2]
		hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
		encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)
		
		batch, channel, height, weight = hidden_states.shape
		residual = hidden_states
		
		hidden_states = self.norm(hidden_states)
		if not self.use_linear_projection:
			hidden_states = self.proj_in(hidden_states)
			inner_dim = hidden_states.shape[1]
			hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
		else:
			inner_dim = hidden_states.shape[1]
			hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
			hidden_states = self.proj_in(hidden_states)
		
		# Blocks
		for block in self.transformer_blocks:
			hidden_states = block(
				hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				timestep=timestep,
				video_length=video_length
			)
		
		# Output
		if not self.use_linear_projection:
			hidden_states = (
				hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
			)
			hidden_states = self.proj_out(hidden_states)
		else:
			hidden_states = self.proj_out(hidden_states)
			hidden_states = (
				hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
			)
		
		output = hidden_states + residual
		
		output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
		if not return_dict:
			return (output,)
		
		return Transformer3DModelOutput(sample=output)
	

class BasicTransformerBlock(nn.Module):
	def __init__(
			self,
			dim: int,
			num_attention_heads: int,
			attention_head_dim: int,
			dropout=0.0,
			cross_attention_dim: Optional[int] = None,
			activation_fn: str = "geglu",
			num_embeds_ada_norm: Optional[int] = None,
			attention_bias: bool = False,
			only_cross_attention: bool = False,
			double_self_attention: bool = False,
			upcast_attention: bool = False,
			norm_elementwise_affine: bool = True,
			norm_type: str = "layer_norm",
			final_dropout: bool = False,
	):
		super().__init__()
		self.only_cross_attention = only_cross_attention
		
		self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
		self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
		
		if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
			raise ValueError(
				f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
				f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
			)
		
		# Define 3 blocks. Each block has its own normalization layer.
		# 1. SC-Cross-Attn
		if self.use_ada_layer_norm:
			self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
		elif self.use_ada_layer_norm_zero:
			self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
		else:
			self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
		self.attn1 = SparseCausalAttention(
			query_dim=dim,
			heads=num_attention_heads,
			dim_head=attention_head_dim,
			dropout=dropout,
			bias=attention_bias,
			cross_attention_dim=cross_attention_dim if only_cross_attention else None,
			upcast_attention=upcast_attention,
		)
		
		# 2. Cross-Attn
		if cross_attention_dim is not None or double_self_attention:
			# We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
			# I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
			# the second cross attention block.
			self.norm2 = (
				AdaLayerNorm(dim, num_embeds_ada_norm)
				if self.use_ada_layer_norm
				else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
			)
			self.attn2 = Attention(
				query_dim=dim,
				cross_attention_dim=cross_attention_dim if not double_self_attention else None,
				heads=num_attention_heads,
				dim_head=attention_head_dim,
				dropout=dropout,
				bias=attention_bias,
				upcast_attention=upcast_attention,
			)  # is self-attn if encoder_hidden_states is none
		else:
			self.norm2 = None
			self.attn2 = None
		
		# 3. Temp-Attn
		self.attn_temp = Attention(
			query_dim=dim,
			heads=num_attention_heads,
			dim_head=attention_head_dim,
			dropout=dropout,
			bias=attention_bias,
			upcast_attention=upcast_attention,
		)
		nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
		self.norm_temp = (
				AdaLayerNorm(dim, num_embeds_ada_norm)
				if self.use_ada_layer_norm
				else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
			)
		
		# 4. Feed-forward
		self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
		self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
		
	def forward(
			self,
			hidden_states,
			attention_mask=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			timestep=None,
			video_length=None,
			cross_attention_kwargs=None,
			class_labels=None,
	):
		# Notice that normalization is always applied before the real computation in the following blocks.
		# 1. Self-Attention
		if self.use_ada_layer_norm:
			norm_hidden_states = self.norm1(hidden_states, timestep)
		elif self.use_ada_layer_norm_zero:
			norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
				hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
			)
		else:
			norm_hidden_states = self.norm1(hidden_states)
		
		cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
		attn_output = self.attn1(
			norm_hidden_states,
			encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
			attention_mask=attention_mask,
			video_length=video_length,
			**cross_attention_kwargs,
		)
		if self.use_ada_layer_norm_zero:
			attn_output = gate_msa.unsqueeze(1) * attn_output
		hidden_states = attn_output + hidden_states
		
		# 2. Cross-Attention
		if self.attn2 is not None:
			norm_hidden_states = (
				self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
			)
			# TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
			# prepare attention mask here
			
			attn_output = self.attn2(
				norm_hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				attention_mask=encoder_attention_mask,
				**cross_attention_kwargs,
			)
			hidden_states = attn_output + hidden_states
		
		# 3. Temporal-Attention
		seq_len = hidden_states.shape[1]
		hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
		norm_hidden_states = (
			self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
		)
		hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
		hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=seq_len)
		
		# 4. Feed-forward
		norm_hidden_states = self.norm3(hidden_states)
		
		if self.use_ada_layer_norm_zero:
			norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
		
		ff_output = self.ff(norm_hidden_states)
		
		if self.use_ada_layer_norm_zero:
			ff_output = gate_mlp.unsqueeze(1) * ff_output
		
		hidden_states = ff_output + hidden_states
		
		return hidden_states
	

class SparseCausalAttention(Attention):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._use_memory_efficient_attention_xformers = False
		self._attention_op = None
	
	def set_xformer_attention_op(self, attention_op: Optional[Callable] = None):
		self._attention_op = attention_op
	
	def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, **cross_attention_kwargs):
		batch_size, sequence_length, _ = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)
		
		attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
		
		query = self.to_q(hidden_states)
		
		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif self.norm_cross:
			encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
		
		key = self.to_k(encoder_hidden_states)
		value = self.to_v(encoder_hidden_states)
		
		former_frame_index = torch.arange(video_length) - 1
		former_frame_index[0] = 0
		
		key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
		key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
		key = rearrange(key, "b f d c -> (b f) d c")
		
		value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
		value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
		value = rearrange(value, "b f d c -> (b f) d c")
		
		query = self.head_to_batch_dim(query).contiguous()
		key = self.head_to_batch_dim(key).contiguous()
		value = self.head_to_batch_dim(value).contiguous()
		
		if self._use_memory_efficient_attention_xformers:
			hidden_states = xformers.ops.memory_efficient_attention(
				query, key, value, attn_bias=attention_mask, op=self._attention_op, scale=self.scale
			)
		else:
			hidden_states = F.scaled_dot_product_attention(
				query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
			)
			
		hidden_states = hidden_states.to(query.dtype)
		hidden_states = self.batch_to_head_dim(hidden_states)
		
		# linear proj
		hidden_states = self.to_out[0](hidden_states)
		# dropout
		hidden_states = self.to_out[1](hidden_states)
		return hidden_states
