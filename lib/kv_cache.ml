(* kv_cache.ml - Paged / Ring Buffer KV Cache Management *)

open Types
open Bigarray

let create ~(n_layer : int) ~(max_seq_len : int) ~(n_head_kv : int) ~(head_dim : int) : kv_cache =
  let total_elems = max_seq_len * n_head_kv * head_dim in
  let layers = Array.init n_layer (fun _ ->
    {
      k = Array1.create float32 c_layout total_elems;
      v = Array1.create float32 c_layout total_elems;
      head = 0;
      n = 0;
    }
  ) in
  {
    layers;
    max_seq_len;
    n_head_kv;
    head_dim;
  }

let update_kv (cache : kv_cache) ~(layer : int) ~(n_past : int) ~(k_new : tensor) ~(v_new : tensor) : unit =
  let l = cache.layers.(layer) in
  let n_new = k_new.ne.(1) in
  let head_dim = cache.head_dim in
  let n_head_kv = cache.n_head_kv in

  let buf_k_new = match k_new.data_f32 with Some b -> b | None -> failwith "k_new not f32" in
  let buf_v_new = match v_new.data_f32 with Some b -> b | None -> failwith "v_new not f32" in

  for h = 0 to n_head_kv - 1 do
    for t = 0 to n_new - 1 do
      let pos = (n_past + t) mod cache.max_seq_len in
      let cache_off = (h * cache.max_seq_len + pos) * head_dim in
      let src_off = (h * n_new + t) * head_dim in
      for d = 0 to head_dim - 1 do
        Array1.unsafe_set l.k (cache_off + d) (Array1.unsafe_get buf_k_new (src_off + d));
        Array1.unsafe_set l.v (cache_off + d) (Array1.unsafe_get buf_v_new (src_off + d));
      done;
    done;
  done;
  l.n <- min cache.max_seq_len (n_past + n_new)

let get_k_view (cache : kv_cache) ~(layer : int) ~(n_tokens : int) : tensor =
  let l = cache.layers.(layer) in
  let head_dim = cache.head_dim in
  let n_head_kv = cache.n_head_kv in
  let t = Tensor.create_3d TYPE_F32 head_dim n_tokens n_head_kv in
  let buf_t = match t.data_f32 with Some b -> b | None -> failwith "get_k_view: not f32" in

  for h = 0 to n_head_kv - 1 do
    for pos = 0 to n_tokens - 1 do
      let cache_off = (h * cache.max_seq_len + pos) * head_dim in
      let dst_off = (h * n_tokens + pos) * head_dim in
      for d = 0 to head_dim - 1 do
        Array1.unsafe_set buf_t (dst_off + d) (Array1.unsafe_get l.k (cache_off + d));
      done;
    done;
  done;
  t

let get_v_view (cache : kv_cache) ~(layer : int) ~(n_tokens : int) : tensor =
  let l = cache.layers.(layer) in
  let head_dim = cache.head_dim in
  let n_head_kv = cache.n_head_kv in
  let t = Tensor.create_3d TYPE_F32 head_dim n_tokens n_head_kv in
  let buf_t = match t.data_f32 with Some b -> b | None -> failwith "get_v_view: not f32" in

  for h = 0 to n_head_kv - 1 do
    for pos = 0 to n_tokens - 1 do
      let cache_off = (h * cache.max_seq_len + pos) * head_dim in
      let dst_off = (h * n_tokens + pos) * head_dim in
      for d = 0 to head_dim - 1 do
        Array1.unsafe_set buf_t (dst_off + d) (Array1.unsafe_get l.v (cache_off + d));
      done;
    done;
  done;
  t

let clear (cache : kv_cache) : unit =
  Array.iter (fun l ->
    Array1.fill l.k 0.0;
    Array1.fill l.v 0.0;
    l.head <- 0;
    l.n <- 0;
  ) cache.layers

let shift_context (cache : kv_cache) ~(shift : int) : unit =
  let head_dim = cache.head_dim in
  let n_head_kv = cache.n_head_kv in
  Array.iter (fun l ->
    if l.n > shift then begin
      let remaining = l.n - shift in
      for h = 0 to n_head_kv - 1 do
        for pos = 0 to remaining - 1 do
          let src_off = (h * cache.max_seq_len + (pos + shift)) * head_dim in
          let dst_off = (h * cache.max_seq_len + pos) * head_dim in
          for d = 0 to head_dim - 1 do
            Array1.unsafe_set l.k (dst_off + d) (Array1.unsafe_get l.k (src_off + d));
            Array1.unsafe_set l.v (dst_off + d) (Array1.unsafe_get l.v (src_off + d));
          done;
        done;
      done;
      l.n <- remaining;
    end else begin
      l.n <- 0;
    end
  ) cache.layers

