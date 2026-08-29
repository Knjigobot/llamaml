(* sampler.mli - Advanced Sampling Pipeline with Penalties, Mirostat & GBNF *)

open Types

val sample : sampler_config -> logits:tensor -> prev_tokens:token_id list -> token_id
val apply_repetition_penalties : float array -> token_id list -> float -> float -> float -> int -> unit
val apply_dry_penalties : float array -> token_id list -> float -> float -> int -> int -> unit
val sample_greedy : float array -> token_id
val sample_top_k_top_p : float array -> temp:float -> top_k:int -> top_p:float -> min_p:float -> token_id
