(* kv_cache.mli - Paged / Ring Buffer KV Cache Management *)

open Types

val create : n_layer:int -> max_seq_len:int -> n_head_kv:int -> head_dim:int -> kv_cache
val update_kv : kv_cache -> layer:int -> n_past:int -> k_new:tensor -> v_new:tensor -> unit
val get_k_view : kv_cache -> layer:int -> n_tokens:int -> tensor
val get_v_view : kv_cache -> layer:int -> n_tokens:int -> tensor
val clear : kv_cache -> unit
val shift_context : kv_cache -> shift:int -> unit
