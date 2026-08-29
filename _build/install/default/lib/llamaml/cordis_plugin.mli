(* cordis_plugin.mli - Cordis-OxCaml Spatiotemporal Plugin Integration *)

open Cordis_core
open Types

module type LLAMAML_SERVICE = sig
  val load_model : string -> Model.model_instance
  val generate : prompt:string -> max_tokens:int -> sampler_config -> ?on_token:(string -> unit) -> unit -> string * inference_metrics
  val embed : prompt:string -> float array
  val get_active_model : unit -> Model.model_instance option
end

val service : (module LLAMAML_SERVICE) Service.t

module Plugin : Registry.PLUGIN
