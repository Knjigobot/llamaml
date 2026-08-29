(* cordis_plugin.ml - Cordis-OxCaml Spatiotemporal Plugin Integration *)

open Types

module type LLAMAML_SERVICE = sig
  val load_model : string -> Model.model_instance
  val generate : prompt:string -> max_tokens:int -> sampler_config -> ?on_token:(string -> unit) -> unit -> string * inference_metrics
  val embed : prompt:string -> float array
  val get_active_model : unit -> Model.model_instance option
end

type 'a service_key = { s_name : string }

module Service = struct
  type 'a t = 'a service_key
  let create name : 'a t = { s_name = name }
  let table = Hashtbl.create 16
  let provide _ctx (key : 'a t) (impl : 'a) = Hashtbl.replace table key.s_name (Obj.repr impl)
  let get _ctx (key : 'a t) : 'a option =
    match Hashtbl.find_opt table key.s_name with
    | Some v -> Some (Obj.obj v)
    | None -> None
end

module Context = struct
  type t = { id : string; mutable effects : (unit -> unit) list }
  let create () = { id = "cordis_ctx"; effects = [] }
  let effect ctx ~label:_ (f : unit -> unit -> unit) =
    let cleanup = f () in
    ctx.effects <- cleanup :: ctx.effects;
    ()
end

module Logger = struct
  type t = string
  let create _ctx : t = "llamaml"
  let info _tag msg = Printf.eprintf "[Cordis-Log] %s\n%!" msg
end

module Registry = struct
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

let service : (module LLAMAML_SERVICE) Service.t = Service.create "llamaml"

let active_model : Model.model_instance option ref = ref None

module LlamamlServiceImpl : LLAMAML_SERVICE = struct
  let load_model (path : string) : Model.model_instance =
    let m = Model.load path in
    active_model := Some m;
    m

  let generate ~(prompt : string) ~(max_tokens : int) (cfg : sampler_config) ?(on_token : (string -> unit) option) () =
    match !active_model with
    | Some m -> Model.generate m ~prompt ~max_tokens cfg ?on_token ()
    | None -> failwith "No active model loaded in Llamaml service"

  let embed ~(prompt : string) =
    match !active_model with
    | Some m -> Model.embed m ~prompt
    | None -> failwith "No active model loaded in Llamaml service"

  let get_active_model () = !active_model
end

module Plugin : Registry.PLUGIN = struct
  let name = "llamaml"
  let version = "1.0.0"
  let inject = ["logger"]

  let on_init (ctx : Context.t) =
    let log = Logger.create ctx in
    Logger.info log "Initializing Llamaml Pure OxCaml Tensor & LLM Engine..."

  let on_start (ctx : Context.t) =
    let log = Logger.create ctx in
    Logger.info log "Llamaml Engine starting. Registering spatiotemporal LLM services...";

    (* Provide Llamaml Service to Cordis Context *)
    Service.provide ctx service (module LlamamlServiceImpl);

    (* Register Revertible Disposal Effect for Active Weights and Buffers *)
    ignore (Context.effect ctx ~label:"llamaml_cleanup" (fun () ->
      (fun () ->
        Logger.info log "Llamaml scope unwinding: freeing model weights, KV caches and scratch memory...";
        (match !active_model with
         | Some m -> Kv_cache.clear m.kv_cache
         | None -> ());
        active_model := None
      )
    ));

    Logger.info log "Llamaml successfully registered in Cordis Spatiotemporal Manifold."

  let on_stop (ctx : Context.t) =
    let log = Logger.create ctx in
    Logger.info log "Llamaml plugin stopping."

  let on_dispose (ctx : Context.t) =
    let log = Logger.create ctx in
    Logger.info log "Llamaml plugin disposed."
end
