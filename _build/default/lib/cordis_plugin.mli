(* cordis_plugin.mli - Cordis-OxCaml Spatiotemporal Plugin Integration *)

open Types

module type LLAMAML_SERVICE = sig
  val load_model : string -> Model.model_instance
  val generate : prompt:string -> max_tokens:int -> sampler_config -> ?on_token:(string -> unit) -> unit -> string * inference_metrics
  val embed : prompt:string -> float array
  val get_active_model : unit -> Model.model_instance option
end

type 'a service_key = { s_name : string }

module Service : sig
  type 'a t = 'a service_key
  val create : string -> 'a t
  val provide : 'a -> 'b t -> 'b -> unit
  val get : 'a -> 'b t -> 'b option
end

module Context : sig
  type t = { id : string; mutable effects : (unit -> unit) list }
  val create : unit -> t
  val effect : t -> label:string -> (unit -> unit -> unit) -> unit
end

module Registry : sig
  module type PLUGIN = sig
    val name : string
    val version : string
    val inject : string list
    val on_init : Context.t -> unit
    val on_start : Context.t -> unit
    val on_stop : Context.t -> unit
    val on_dispose : Context.t -> unit
  end
end

val service : (module LLAMAML_SERVICE) Service.t

module Plugin : Registry.PLUGIN
