(* types.mli - Type Definitions for Pure OxCaml GGML Tensor & LLaMA.cpp Engine *)

open Bigarray

type quant_type =
  | TYPE_F32
  | TYPE_F16
  | TYPE_Q4_0
  | TYPE_Q4_1
  | TYPE_Q5_0
  | TYPE_Q5_1
  | TYPE_Q8_0
  | TYPE_Q8_1
  | TYPE_Q2_K
  | TYPE_Q3_K
  | TYPE_Q4_K
  | TYPE_Q5_K
  | TYPE_Q6_K
  | TYPE_Q8_K
  | TYPE_IQ2_XXS
  | TYPE_IQ2_XS
  | TYPE_IQ3_XXS
  | TYPE_IQ1_S
  | TYPE_IQ4_NL
  | TYPE_IQ4_XS
  | TYPE_BF16

type op_type =
  | OP_NONE
  | OP_DUP
  | OP_ADD
  | OP_ADD1
  | OP_ACC
  | OP_SUB
  | OP_MUL
  | OP_DIV
  | OP_SQR
  | OP_SQRT
  | OP_LOG
  | OP_EXP
  | OP_SIN
  | OP_COS
  | OP_SILU
  | OP_GELU
  | OP_GELU_QUICK
  | OP_RELU
  | OP_SIGMOID
  | OP_TANH
  | OP_SWIGLU
  | OP_NORM
  | OP_RMS_NORM
  | OP_GROUP_NORM
  | OP_MUL_MAT
  | OP_MUL_MAT_ID
  | OP_OUT_PROD
  | OP_SCALE
  | OP_SET
  | OP_CPY
  | OP_CONT
  | OP_RESHAPE
  | OP_VIEW
  | OP_PERMUTE
  | OP_TRANSPOSE
  | OP_GET_ROWS
  | OP_GET_ROWS_BACK
  | OP_DIAG
  | OP_DIAG_MASK_INF
  | OP_DIAG_MASK_ZERO
  | OP_SOFT_MAX
  | OP_SOFT_MAX_EXT
  | OP_ROPE
  | OP_ROPE_BACK
  | OP_ALIBI
  | OP_FLASH_ATTN_EXT
  | OP_FLASH_FFN
  | OP_CROSS_ENTROPY_LOSS
  | OP_CONCAT
  | OP_ARGSORT
  | OP_SUM_ROWS
  | OP_UPSCALE
  | OP_PAD
  | OP_TIMESTEP_EMBEDDING
  | OP_UNRAVEL
  | OP_MOE_GATE

type f32_buffer = (float, float32_elt, c_layout) Array1.t
type u8_buffer = (int, int8_unsigned_elt, c_layout) Array1.t

type tensor = {
  id : int;
  name : string;
  qtype : quant_type;
  ne : int array;   (* Dimensions [d0, d1, d2, d3] *)
  nb : int array;   (* Byte strides [nb0, nb1, nb2, nb3] *)
  n_dims : int;
  op : op_type;
  op_params : int array;
  mutable flags : int;
  src0 : tensor option;
  src1 : tensor option;
  src2 : tensor option;
  mutable data_f32 : f32_buffer option;
  mutable data_raw : u8_buffer option;
  mutable grad : tensor option;
}

type cgraph = {
  mutable nodes : tensor list;
  mutable leafs : tensor list;
  mutable n_nodes : int;
  mutable n_leafs : int;
}

type arch_type =
  | ARCH_LLAMA
  | ARCH_MISTRAL
  | ARCH_MIXTRAL
  | ARCH_QWEN2
  | ARCH_DEEPSEEK2
  | ARCH_GEMMA
  | ARCH_GEMMA2
  | ARCH_PHI3
  | ARCH_FALCON
  | ARCH_STARCODER
  | ARCH_COMMAND_R
  | ARCH_UNKNOWN of string

type hyperparameters = {
  arch : arch_type;
  n_vocab : int;
  n_embd : int;
  n_mult : int;
  n_head : int;
  n_head_kv : int;
  n_layer : int;
  n_rot : int;
  n_expert : int;
  n_expert_used : int;
  n_ff : int;
  rope_freq_base : float;
  rope_freq_scale : float;
  rope_type : int;
  rms_norm_eps : float;
  f_clamp_kqv : float;
  f_max_alibi_bias : float;
  expert_weights_scale : float;
}

type token_id = int

type tokenizer_type =
  | TOKENIZER_BPE
  | TOKENIZER_SPM
  | TOKENIZER_WPM
  | TOKENIZER_UGM

type token_data = {
  text : string;
  score : float;
  token_type : int;
}

type tokenizer = {
  t_type : tokenizer_type;
  vocab : (token_id, token_data) Hashtbl.t;
  token_to_id : (string, token_id) Hashtbl.t;
  merges : (string * string, int) Hashtbl.t;
  bos_id : token_id option;
  eos_id : token_id option;
  pad_id : token_id option;
  nl_id : token_id option;
  unk_id : token_id option;
  add_bos : bool;
  add_eos : bool;
}

type layer_weights = {
  attn_norm : tensor option;
  attn_norm_2 : tensor option;
  wq : tensor option;
  wk : tensor option;
  wv : tensor option;
  wo : tensor option;
  ffn_norm : tensor option;
  ffn_norm_2 : tensor option;
  w_gate : tensor option;
  w_up : tensor option;
  w_down : tensor option;
  w_gate_exp : tensor list;
  w_up_exp : tensor list;
  w_down_exp : tensor list;
  mla_q_lora : tensor option;
  mla_kv_lora : tensor option;
}

type model_weights = {
  token_embd : tensor option;
  output_norm : tensor option;
  output_head : tensor option;
  layers : layer_weights array;
}

type sampler_config = {
  temperature : float;
  top_k : int;
  top_p : float;
  min_p : float;
  typical_p : float;
  tfs_z : float;
  repeat_penalty : float;
  frequency_penalty : float;
  presence_penalty : float;
  repeat_last_n : int;
  dry_multiplier : float;
  dry_base : float;
  dry_allowed_length : int;
  dry_penalty_last_n : int;
  mirostat_mode : int;
  mirostat_tau : float;
  mirostat_eta : float;
  grammar_gbnf : string option;
  logit_bias : (token_id, float) Hashtbl.t;
}

type kv_cache_layer = {
  k : f32_buffer;
  v : f32_buffer;
  mutable head : int;
  mutable n : int;
}

type kv_cache = {
  layers : kv_cache_layer array;
  max_seq_len : int;
  n_head_kv : int;
  head_dim : int;
}

type inference_metrics = {
  prompt_tokens : int;
  gen_tokens : int;
  prompt_eval_ms : float;
  gen_ms : float;
  tokens_per_sec : float;
  peak_memory_mb : float;
}

val type_name : quant_type -> string
val type_block_size : quant_type -> int
val type_type_size : quant_type -> int
val op_name : op_type -> string
val arch_to_string : arch_type -> string
val string_to_arch : string -> arch_type
val default_hyperparameters : hyperparameters
val default_sampler_config : sampler_config
