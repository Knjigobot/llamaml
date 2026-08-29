(* tokenizer.mli - High-Fidelity BPE & SentencePiece Tokenizer *)

open Types

val create : ?t_type:tokenizer_type -> unit -> tokenizer
val add_token : tokenizer -> token_id -> string -> float -> int -> unit
val add_merge : tokenizer -> string -> string -> int -> unit

val encode : tokenizer -> ?bos:bool -> ?eos:bool -> string -> token_id array
val decode : tokenizer -> token_id array -> string

val token_to_piece : tokenizer -> token_id -> string
val piece_to_token : tokenizer -> string -> token_id option
val is_special_token : tokenizer -> token_id -> bool
