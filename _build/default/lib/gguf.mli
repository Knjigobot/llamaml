(* gguf.mli - Complete Binary GGUF v2/v3 Deserializer & Metadata Inspector *)

open Types
open Bigarray

type metadata_val =
  | G_UINT8 of int
  | G_INT8 of int
  | G_UINT16 of int
  | G_INT16 of int
  | G_UINT32 of int32
  | G_INT32 of int32
  | G_FLOAT32 of float
  | G_BOOL of bool
  | G_STRING of string
  | G_ARRAY of metadata_val list
  | G_UINT64 of int64
  | G_INT64 of int64
  | G_FLOAT64 of float

type tensor_info = {
  ti_name : string;
  ti_n_dims : int;
  ti_ne : int array;
  ti_qtype : quant_type;
  ti_offset : int64;
}

type gguf_file = {
  version : int;
  metadata : (string, metadata_val) Hashtbl.t;
  tensor_infos : (string, tensor_info) Hashtbl.t;
  tensor_list : tensor_info list;
  data_offset : int64;
  file_path : string;
  raw_buffer : u8_buffer option;
}

val parse_file : string -> gguf_file
val get_metadata_string : gguf_file -> string -> string option
val get_metadata_int : gguf_file -> string -> int option
val get_metadata_float : gguf_file -> string -> float option
val get_metadata_array : gguf_file -> string -> metadata_val list option

val extract_hyperparameters : gguf_file -> hyperparameters
val extract_tokenizer : gguf_file -> tokenizer
val load_tensor : gguf_file -> string -> tensor
val load_model_weights : gguf_file -> hyperparameters -> model_weights
