(* alloc.mli - Virtual Memory Planning & Scratch Buffer Allocator *)

open Types
open Bigarray

type memory_plan = {
  total_size_bytes : int;
  tensor_offsets : (int, int) Hashtbl.t;
}

type scratch_pool = {
  buffer : f32_buffer;
  mutable current_offset : int;
  max_size : int;
}

val plan_graph_memory : cgraph -> memory_plan
val create_scratch_pool : int -> scratch_pool
val scratch_alloc : scratch_pool -> int -> f32_buffer
val scratch_reset : scratch_pool -> unit
