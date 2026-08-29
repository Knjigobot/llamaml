(* server.mli - OpenAI-Compatible HTTP / SSE Inference Daemon *)

open Types

type server_config = {
  port : int;
  host : string;
  model_path : string;
  static_dir : string option;
}

type t

val start : server_config -> t
val stop : t -> unit
val run_forever : t -> unit
