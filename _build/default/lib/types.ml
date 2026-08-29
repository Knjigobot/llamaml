(* types.ml - Type Implementation for Pure OxCaml GGML Tensor & LLaMA.cpp Engine *)

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
  ne : int array;
  nb : int array;
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
  attn_q_norm : tensor option;
  attn_k_norm : tensor option;
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

let type_name = function
  | TYPE_F32 -> "F32"
  | TYPE_F16 -> "F16"
  | TYPE_Q4_0 -> "Q4_0"
  | TYPE_Q4_1 -> "Q4_1"
  | TYPE_Q5_0 -> "Q5_0"
  | TYPE_Q5_1 -> "Q5_1"
  | TYPE_Q8_0 -> "Q8_0"
  | TYPE_Q8_1 -> "Q8_1"
  | TYPE_Q2_K -> "Q2_K"
  | TYPE_Q3_K -> "Q3_K"
  | TYPE_Q4_K -> "Q4_K"
  | TYPE_Q5_K -> "Q5_K"
  | TYPE_Q6_K -> "Q6_K"
  | TYPE_Q8_K -> "Q8_K"
  | TYPE_IQ2_XXS -> "IQ2_XXS"
  | TYPE_IQ2_XS -> "IQ2_XS"
  | TYPE_IQ3_XXS -> "IQ3_XXS"
  | TYPE_IQ1_S -> "IQ1_S"
  | TYPE_IQ4_NL -> "IQ4_NL"
  | TYPE_IQ4_XS -> "IQ4_XS"
  | TYPE_BF16 -> "BF16"

let type_block_size = function
  | TYPE_F32 -> 1
  | TYPE_F16 -> 1
  | TYPE_BF16 -> 1
  | TYPE_Q4_0 -> 32
  | TYPE_Q4_1 -> 32
  | TYPE_Q5_0 -> 32
  | TYPE_Q5_1 -> 32
  | TYPE_Q8_0 -> 32
  | TYPE_Q8_1 -> 32
  | TYPE_Q2_K -> 256
  | TYPE_Q3_K -> 256
  | TYPE_Q4_K -> 256
  | TYPE_Q5_K -> 256
  | TYPE_Q6_K -> 256
  | TYPE_Q8_K -> 256
  | TYPE_IQ2_XXS -> 256
  | TYPE_IQ2_XS -> 256
  | TYPE_IQ3_XXS -> 256
  | TYPE_IQ1_S -> 256
  | TYPE_IQ4_NL -> 32
  | TYPE_IQ4_XS -> 256

let type_type_size = function
  | TYPE_F32 -> 4
  | TYPE_F16 -> 2
  | TYPE_BF16 -> 2
  | TYPE_Q4_0 -> 18     (* 2 bytes f16 delta + 16 bytes for 32 nibbles *)
  | TYPE_Q4_1 -> 20     (* 2 bytes f16 delta + 2 bytes f16 min + 16 bytes *)
  | TYPE_Q5_0 -> 22     (* 2 bytes f16 delta + 4 bytes high bits + 16 bytes *)
  | TYPE_Q5_1 -> 24     (* 2 bytes f16 delta + 2 bytes f16 min + 4 bytes + 16 bytes *)
  | TYPE_Q8_0 -> 34     (* 2 bytes f16 delta + 32 bytes int8 *)
  | TYPE_Q8_1 -> 36     (* 4 bytes delta/min + 32 bytes int8 *)
  | TYPE_Q2_K -> 256 * 2 / 8 + 64
  | TYPE_Q3_K -> 256 * 3 / 8 + 64
  | TYPE_Q4_K -> 144    (* 2 bytes f16 d + 2 bytes f16 dmin + 12 bytes scales + 128 bytes qs *)
  | TYPE_Q5_K -> 176    (* 2 bytes f16 d + 2 bytes f16 dmin + 12 bytes scales + 32 bytes high + 128 bytes *)
  | TYPE_Q6_K -> 210    (* 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d *)
  | TYPE_Q8_K -> 256 + 32
  | TYPE_IQ2_XXS -> 66
  | TYPE_IQ2_XS -> 74
  | TYPE_IQ3_XXS -> 98
  | TYPE_IQ1_S -> 50
  | TYPE_IQ4_NL -> 18
  | TYPE_IQ4_XS -> 136

let op_name = function
  | OP_NONE -> "NONE"
  | OP_DUP -> "DUP"
  | OP_ADD -> "ADD"
  | OP_ADD1 -> "ADD1"
  | OP_ACC -> "ACC"
  | OP_SUB -> "SUB"
  | OP_MUL -> "MUL"
  | OP_DIV -> "DIV"
  | OP_SQR -> "SQR"
  | OP_SQRT -> "SQRT"
  | OP_LOG -> "LOG"
  | OP_EXP -> "EXP"
  | OP_SIN -> "SIN"
  | OP_COS -> "COS"
  | OP_SILU -> "SILU"
  | OP_GELU -> "GELU"
  | OP_GELU_QUICK -> "GELU_QUICK"
  | OP_RELU -> "RELU"
  | OP_SIGMOID -> "SIGMOID"
  | OP_TANH -> "TANH"
  | OP_SWIGLU -> "SWIGLU"
  | OP_NORM -> "NORM"
  | OP_RMS_NORM -> "RMS_NORM"
  | OP_GROUP_NORM -> "GROUP_NORM"
  | OP_MUL_MAT -> "MUL_MAT"
  | OP_MUL_MAT_ID -> "MUL_MAT_ID"
  | OP_OUT_PROD -> "OUT_PROD"
  | OP_SCALE -> "SCALE"
  | OP_SET -> "SET"
  | OP_CPY -> "CPY"
  | OP_CONT -> "CONT"
  | OP_RESHAPE -> "RESHAPE"
  | OP_VIEW -> "VIEW"
  | OP_PERMUTE -> "PERMUTE"
  | OP_TRANSPOSE -> "TRANSPOSE"
  | OP_GET_ROWS -> "GET_ROWS"
  | OP_GET_ROWS_BACK -> "GET_ROWS_BACK"
  | OP_DIAG -> "DIAG"
  | OP_DIAG_MASK_INF -> "DIAG_MASK_INF"
  | OP_DIAG_MASK_ZERO -> "DIAG_MASK_ZERO"
  | OP_SOFT_MAX -> "SOFT_MAX"
  | OP_SOFT_MAX_EXT -> "SOFT_MAX_EXT"
  | OP_ROPE -> "ROPE"
  | OP_ROPE_BACK -> "ROPE_BACK"
  | OP_ALIBI -> "ALIBI"
  | OP_FLASH_ATTN_EXT -> "FLASH_ATTN_EXT"
  | OP_FLASH_FFN -> "FLASH_FFN"
  | OP_CROSS_ENTROPY_LOSS -> "CROSS_ENTROPY_LOSS"
  | OP_CONCAT -> "CONCAT"
  | OP_ARGSORT -> "ARGSORT"
  | OP_SUM_ROWS -> "SUM_ROWS"
  | OP_UPSCALE -> "UPSCALE"
  | OP_PAD -> "PAD"
  | OP_TIMESTEP_EMBEDDING -> "TIMESTEP_EMBEDDING"
  | OP_UNRAVEL -> "UNRAVEL"
  | OP_MOE_GATE -> "MOE_GATE"

let arch_to_string = function
  | ARCH_LLAMA -> "llama"
  | ARCH_MISTRAL -> "mistral"
  | ARCH_MIXTRAL -> "mixtral"
  | ARCH_QWEN2 -> "qwen2"
  | ARCH_DEEPSEEK2 -> "deepseek2"
  | ARCH_GEMMA -> "gemma"
  | ARCH_GEMMA2 -> "gemma2"
  | ARCH_PHI3 -> "phi3"
  | ARCH_FALCON -> "falcon"
  | ARCH_STARCODER -> "starcoder"
  | ARCH_COMMAND_R -> "command-r"
  | ARCH_UNKNOWN s -> s

let string_to_arch = function
  | "llama" | "llama2" | "llama3" -> ARCH_LLAMA
  | "mistral" -> ARCH_MISTRAL
  | "mixtral" -> ARCH_MIXTRAL
  | "qwen2" | "qwen2.5" | "qwen3" -> ARCH_QWEN2
  | "deepseek2" | "deepseek" -> ARCH_DEEPSEEK2
  | "gemma" -> ARCH_GEMMA
  | "gemma2" -> ARCH_GEMMA2
  | "phi3" | "phi" -> ARCH_PHI3
  | "falcon" -> ARCH_FALCON
  | "starcoder" | "starcoder2" -> ARCH_STARCODER
  | "command-r" -> ARCH_COMMAND_R
  | s -> ARCH_UNKNOWN s

let default_hyperparameters = {
  arch = ARCH_LLAMA;
  n_vocab = 32000;
  n_embd = 4096;
  n_mult = 256;
  n_head = 32;
  n_head_kv = 32;
  n_layer = 32;
  n_rot = 128;
  n_expert = 0;
  n_expert_used = 0;
  n_ff = 11008;
  rope_freq_base = 10000.0;
  rope_freq_scale = 1.0;
  rope_type = 0;
  rms_norm_eps = 1e-5;
  f_clamp_kqv = 0.0;
  f_max_alibi_bias = 0.0;
  expert_weights_scale = 1.0;
}

let default_sampler_config = {
  temperature = 0.7;
  top_k = 40;
  top_p = 0.9;
  min_p = 0.05;
  typical_p = 1.0;
  tfs_z = 1.0;
  repeat_penalty = 1.1;
  frequency_penalty = 0.0;
  presence_penalty = 0.0;
  repeat_last_n = 64;
  dry_multiplier = 0.0;
  dry_base = 1.75;
  dry_allowed_length = 2;
  dry_penalty_last_n = -1;
  mirostat_mode = 0;
  mirostat_tau = 5.0;
  mirostat_eta = 0.1;
  grammar_gbnf = None;
  logit_bias = Hashtbl.create 8;
}
