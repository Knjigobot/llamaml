(* alloc.ml - Virtual Memory Planning & Scratch Buffer Allocator *)

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

let plan_graph_memory (g : cgraph) : memory_plan =
  let offsets = Hashtbl.create 64 in
  let current_offset = ref 0 in
  List.iter (fun (t : tensor) ->
    let size = Tensor.nbytes t in
    let aligned_size = (size + 63) land (lnot 63) in
    Hashtbl.add offsets t.id !current_offset;
    current_offset := !current_offset + aligned_size;
  ) g.nodes;
  {
    total_size_bytes = !current_offset;
    tensor_offsets = offsets;
  }

let create_scratch_pool (size_elements : int) : scratch_pool =
  {
    buffer = Array1.create float32 c_layout size_elements;
    current_offset = 0;
    max_size = size_elements;
  }

let scratch_alloc (pool : scratch_pool) (n_elements : int) : f32_buffer =
  if pool.current_offset + n_elements > pool.max_size then
    failwith "Scratch buffer overflow";
  let sub = Array1.sub pool.buffer pool.current_offset n_elements in
  pool.current_offset <- pool.current_offset + n_elements;
  sub

let scratch_reset (pool : scratch_pool) : unit =
  pool.current_offset <- 0
