#!/usr/bin/env python3
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.# Install dependencies for Google Colab.
# If you want to run this notebook on your own machine, you can skip this cell

from typing import Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pickle

from perceiver import perceiver, position_encoding, io_processors, bytes_tokenizer

###################################
# Load parameters from checkpoint #
###################################

with open("perceiver_io/language_perceiver_io_bytes.pickle", "rb") as f:
  params = pickle.loads(f.read())

################
# Model config #
################

D_MODEL = 768
D_LATENTS = 1280
MAX_SEQ_LEN = 2048

encoder_config = dict(
    num_self_attends_per_block=26,
    num_blocks=1,
    z_index_dim=256,
    num_z_channels=D_LATENTS,
    num_self_attend_heads=8,
    num_cross_attend_heads=8,
    qk_channels=8 * 32,
    v_channels=D_LATENTS,
    use_query_residual=True,
    cross_attend_widening_factor=1,
    self_attend_widening_factor=1)

decoder_config = dict(
    output_num_channels=D_LATENTS,
    position_encoding_type='trainable',
    output_index_dims=MAX_SEQ_LEN,
    num_z_channels=D_LATENTS,
    qk_channels=8 * 32,
    v_channels=D_MODEL,
    num_heads=8,
    final_project=False,
    use_query_residual=False,
    trainable_position_encoding_kwargs=dict(num_channels=D_MODEL))

# The tokenizer is just UTF-8 encoding (with an offset)
tokenizer = bytes_tokenizer.BytesTokenizer()

############################
# Decoding Perceiver Model #
############################

def apply_perceiver(
    inputs: jnp.ndarray, input_mask: jnp.ndarray) -> jnp.ndarray:
  """Runs a forward pass on the Perceiver.

  Args:
    inputs: input bytes, an int array of shape [B, T]
    input_mask: Array of shape indicating which entries are valid and which are
      masked. A truthy value indicates that the entry is valid.

  Returns:
    The output logits, an array of shape [B, T, vocab_size].
  """
  assert inputs.shape[1] == MAX_SEQ_LEN

  embedding_layer = hk.Embed(
      vocab_size=tokenizer.vocab_size,
      embed_dim=D_MODEL)
  embedded_inputs = embedding_layer(inputs)

  batch_size = embedded_inputs.shape[0]

  input_pos_encoding = perceiver.position_encoding.TrainablePositionEncoding(
      index_dim=MAX_SEQ_LEN, num_channels=D_MODEL)
  embedded_inputs = embedded_inputs + input_pos_encoding(batch_size)
  perceiver_mod = perceiver.Perceiver(
      encoder=perceiver.PerceiverEncoder(**encoder_config),
      decoder=perceiver.BasicDecoder(**decoder_config))
  output_embeddings = perceiver_mod(
      embedded_inputs, is_training=False, input_mask=input_mask, query_mask=input_mask)

  logits = io_processors.EmbeddingDecoder(
      embedding_matrix=embedding_layer.embeddings)(output_embeddings)
  return logits

apply_perceiver = hk.transform(apply_perceiver).apply

input_str = "This is an incomplete sentence where some words are missing."
input_tokens = tokenizer.to_int(input_str)

# Mask " missing.". Note that the model performs much better if the masked chunk
# starts with a space.
#input_tokens[51:60] = tokenizer.mask_token
input_tokens[input_str.find(" incomplete"):60] = tokenizer.mask_token
print("Tokenized string without masked bytes:")
print(tokenizer.to_string(input_tokens))

##########################
# Pad and reshape inputs #
##########################

inputs = input_tokens[None]
input_mask = np.ones_like(inputs)

def pad(max_sequence_length: int, inputs, input_mask):
  input_len = inputs.shape[1]
  assert input_len <= max_sequence_length
  pad_len = max_sequence_length - input_len
  padded_inputs = np.pad(
      inputs,
      pad_width=((0, 0), (0, pad_len)),
      constant_values=tokenizer.pad_token)
  padded_mask = np.pad(
      input_mask,
      pad_width=((0, 0), (0, pad_len)),
      constant_values=0)
  return padded_inputs, padded_mask

inputs, input_mask = pad(MAX_SEQ_LEN, inputs, input_mask)

rng = jax.random.PRNGKey(1)  # Unused

out = apply_perceiver(params, rng=rng, inputs=inputs, input_mask=input_mask)

masked_tokens_predictions = out[0, 51:60].argmax(axis=-1)
print("Greedy predictions:")
print(masked_tokens_predictions)
print()
print("Predicted string:")
print(tokenizer.to_string(masked_tokens_predictions))
