(* tensor.ml - Multidimensional Bigarray Tensor Manipulation *)

open Types
open Bigarray

let id_counter = ref 0

let next_id () =
  incr id_counter;
  !id_counter

let compute_byte_strides (qtype : quant_type) (ne : int array) : int array =
  let ts = type_type_size qtype in
  let bs = type_block_size qtype in
  let nb = Array.make 4 0 in
  nb.(0) <- ts;
  nb.(1) <- nb.(0) * (ne.(0) / bs);
  nb.(2) <- nb.(1) * ne.(1);
  nb.(3) <- nb.(2) * ne.(2);
  nb

let nelements (t : tensor) : int =
  t.ne.(0) * t.ne.(1) * t.ne.(2) * t.ne.(3)

let nbytes (t : tensor) : int =
  let bs = type_block_size t.qtype in
  let ts = type_type_size t.qtype in
  let n_blocks = nelements t / bs in
  n_blocks * ts

let create_tensor ?(name = "") (qtype : quant_type) (ne : int array) (n_dims : int) : tensor =
  let nb = compute_byte_strides qtype ne in
  let total_elems = ne.(0) * ne.(1) * ne.(2) * ne.(3) in
  let bs = type_block_size qtype in
  let ts = type_type_size qtype in
  let total_bytes = (total_elems / bs) * ts in

  let data_f32, data_raw =
    if qtype = TYPE_F32 then
      (Some (Array1.create float32 c_layout total_elems), None)
    else
      (None, Some (Array1.create int8_unsigned c_layout total_bytes))
  in

  {
    id = next_id ();
    name = if name = "" then Printf.sprintf "tensor_%d" !id_counter else name;
    qtype;
    ne;
    nb;
    n_dims;
    op = OP_NONE;
    op_params = [||];
    flags = 0;
    src0 = None;
    src1 = None;
    src2 = None;
    data_f32;
    data_raw;
    grad = None;
  }

let create ?name qtype ne =
  create_tensor ?name qtype ne (Array.length ne)

let create_1d ?name qtype ne0 =
  create_tensor ?name qtype [| ne0; 1; 1; 1 |] 1

let create_2d ?name qtype ne0 ne1 =
  create_tensor ?name qtype [| ne0; ne1; 1; 1 |] 2

let create_3d ?name qtype ne0 ne1 ne2 =
  create_tensor ?name qtype [| ne0; ne1; ne2; 1 |] 3

let create_4d ?name qtype ne0 ne1 ne2 ne3 =
  create_tensor ?name qtype [| ne0; ne1; ne2; ne3 |] 4

let from_f32_array ?name (dims : int array) (data : float array) : tensor =
  let ne = [| 1; 1; 1; 1 |] in
  for i = 0 to min (Array.length dims - 1) 3 do
    ne.(i) <- dims.(i);
  done;
  let t = create_tensor ?name TYPE_F32 ne (Array.length dims) in
  match t.data_f32 with
  | Some buf ->
    let n = min (Array1.dim buf) (Array.length data) in
    for i = 0 to n - 1 do
      Array1.unsafe_set buf i data.(i);
    done;
    t
  | None -> failwith "Failed to allocate f32 tensor buffer"

let to_f32_array (t : tensor) : float array =
  let n = nelements t in
  let arr = Array.make n 0.0 in
  match t.data_f32 with
  | Some buf ->
    for i = 0 to n - 1 do
      arr.(i) <- Array1.unsafe_get buf i;
    done;
    arr
  | None ->
    match t.data_raw with
    | Some raw ->
      let temp = Array1.create float32 c_layout n in
      (match t.qtype with
       | TYPE_F16 -> Quant.dequantize_row_f16 raw 0 temp 0 n
       | TYPE_Q4_0 -> Quant.dequantize_row_q4_0 raw 0 temp 0 n
       | TYPE_Q4_1 -> Quant.dequantize_row_q4_1 raw 0 temp 0 n
       | TYPE_Q8_0 -> Quant.dequantize_row_q8_0 raw 0 temp 0 n
       | TYPE_Q4_K -> Quant.dequantize_row_q4_k raw 0 temp 0 n
       | TYPE_Q5_K -> Quant.dequantize_row_q5_k raw 0 temp 0 n
       | TYPE_Q6_K -> Quant.dequantize_row_q6_k raw 0 temp 0 n
       | _ -> failwith "Unsupported dequantization format");
      for i = 0 to n - 1 do
        arr.(i) <- Array1.unsafe_get temp i;
      done;
      arr
    | None -> arr

let get_f32_1d (t : tensor) (i0 : int) : float =
  match t.data_f32 with
  | Some buf -> Array1.unsafe_get buf i0
  | None -> failwith "Tensor not in f32 format"

let set_f32_1d (t : tensor) (i0 : int) (v : float) : unit =
  match t.data_f32 with
  | Some buf -> Array1.unsafe_set buf i0 v
  | None -> failwith "Tensor not in f32 format"

let get_f32_2d (t : tensor) (i0 : int) (i1 : int) : float =
  let idx = i1 * t.ne.(0) + i0 in
  get_f32_1d t idx

let set_f32_2d (t : tensor) (i0 : int) (i1 : int) (v : float) : unit =
  let idx = i1 * t.ne.(0) + i0 in
  set_f32_1d t idx v

let view_1d (t : tensor) (offset : int) (ne0 : int) : tensor =
  let sub_f32 = match t.data_f32 with
    | Some buf -> Some (Array1.sub buf offset ne0)
    | None -> None
  in
  let bs = type_block_size t.qtype in
  let ts = type_type_size t.qtype in
  let sub_raw = match t.data_raw with
    | Some buf ->
      let byte_off = (offset / bs) * ts in
      let byte_len = (ne0 / bs) * ts in
      Some (Array1.sub buf byte_off byte_len)
    | None -> None
  in
  {
    id = next_id ();
    name = Printf.sprintf "%s_view1d" t.name;
    qtype = t.qtype;
    ne = [| ne0; 1; 1; 1 |];
    nb = compute_byte_strides t.qtype [| ne0; 1; 1; 1 |];
    n_dims = 1;
    op = OP_VIEW;
    op_params = [| offset |];
    flags = 0;
    src0 = Some t;
    src1 = None;
    src2 = None;
    data_f32 = sub_f32;
    data_raw = sub_raw;
    grad = None;
  }

let view_2d (t : tensor) (offset : int) (ne0 : int) (ne1 : int) (stride1 : int) : tensor =
  let total_elems = ne1 * stride1 in
  let sub_f32 = match t.data_f32 with
    | Some buf -> Some (Array1.sub buf offset total_elems)
    | None -> None
  in
  {
    id = next_id ();
    name = Printf.sprintf "%s_view2d" t.name;
    qtype = t.qtype;
    ne = [| ne0; ne1; 1; 1 |];
    nb = [| 4; stride1 * 4; stride1 * ne1 * 4; stride1 * ne1 * 4 |];
    n_dims = 2;
    op = OP_VIEW;
    op_params = [| offset; stride1 |];
    flags = 0;
    src0 = Some t;
    src1 = None;
    src2 = None;
    data_f32 = sub_f32;
    data_raw = None;
    grad = None;
  }

let reshape (t : tensor) (dims : int array) : tensor =
  let ne = [| 1; 1; 1; 1 |] in
  for i = 0 to min (Array.length dims - 1) 3 do
    ne.(i) <- dims.(i);
  done;
  {
    id = next_id ();
    name = Printf.sprintf "%s_reshape" t.name;
    qtype = t.qtype;
    ne;
    nb = compute_byte_strides t.qtype ne;
    n_dims = Array.length dims;
    op = OP_RESHAPE;
    op_params = [||];
    flags = 0;
    src0 = Some t;
    src1 = None;
    src2 = None;
    data_f32 = t.data_f32;
    data_raw = t.data_raw;
    grad = None;
  }

let copy_f32 (src : tensor) (dst : tensor) : unit =
  match (src.data_f32, dst.data_f32) with
  | Some s_buf, Some d_buf ->
    Array1.blit s_buf d_buf
  | _ -> failwith "copy_f32 requires both tensors to have f32 data"

let zero (t : tensor) : unit =
  match t.data_f32 with
  | Some buf -> Array1.fill buf 0.0
  | None ->
    match t.data_raw with
    | Some buf -> Array1.fill buf 0
    | None -> ()
