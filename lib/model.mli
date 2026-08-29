(* model.mli - Universal Transformer Forward & Generation Engine *)

open Types

type model_instance = {
  hp : hyperparameters;
  weights : model_weights;
  tokenizer : tokenizer;
  kv_cache : kv_cache;
  gguf : gguf_file;
}

val load : string -> model_instance
val forward : model_instance -> token_ids:token_id array -> n_past:int -> tensor
val generate : model_instance -> prompt:string -> max_tokens:int -> sampler_config -> ?on_token:(string -> unit) -> unit -> string * inference_metrics
val embed : model_instance -> prompt:string -> float array
