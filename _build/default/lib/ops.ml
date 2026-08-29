(* ops.ml - Mathematical Tensor Operators & High-Performance Kernels *)

open Types
open Bigarray

let add (a : tensor) (b : tensor) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "add: a not f32" in
  let buf_b = match b.data_f32 with Some buf -> buf | None -> failwith "add: b not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "add: out not f32" in
  for i = 0 to n - 1 do
    Array1.unsafe_set buf_out i (Array1.unsafe_get buf_a i +. Array1.unsafe_get buf_b i);
  done;
  out

let sub (a : tensor) (b : tensor) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "sub: a not f32" in
  let buf_b = match b.data_f32 with Some buf -> buf | None -> failwith "sub: b not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "sub: out not f32" in
  for i = 0 to n - 1 do
    Array1.unsafe_set buf_out i (Array1.unsafe_get buf_a i -. Array1.unsafe_get buf_b i);
  done;
  out

let mul (a : tensor) (b : tensor) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "mul: a not f32" in
  let buf_b = match b.data_f32 with Some buf -> buf | None -> failwith "mul: b not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "mul: out not f32" in
  for i = 0 to n - 1 do
    Array1.unsafe_set buf_out i (Array1.unsafe_get buf_a i *. Array1.unsafe_get buf_b i);
  done;
  out

let scale (a : tensor) (s : float) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "scale: a not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "scale: out not f32" in
  for i = 0 to n - 1 do
    Array1.unsafe_set buf_out i (Array1.unsafe_get buf_a i *. s);
  done;
  out

let silu (a : tensor) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "silu: a not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "silu: out not f32" in
  for i = 0 to n - 1 do
    let x = Array1.unsafe_get buf_a i in
    let sigm = 1.0 /. (1.0 +. exp (-. x)) in
    Array1.unsafe_set buf_out i (x *. sigm);
  done;
  out

let gelu (a : tensor) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "gelu: a not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "gelu: out not f32" in
  let c = sqrt (2.0 /. Float.pi) in
  for i = 0 to n - 1 do
    let x = Array1.unsafe_get buf_a i in
    let inner = c *. (x +. 0.044715 *. x *. x *. x) in
    let y = 0.5 *. x *. (1.0 +. tanh inner) in
    Array1.unsafe_set buf_out i y;
  done;
  out

let gelu_quick (a : tensor) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "gelu_quick: a not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "gelu_quick: out not f32" in
  for i = 0 to n - 1 do
    let x = Array1.unsafe_get buf_a i in
    let sigm = 1.0 /. (1.0 +. exp (-. (1.702 *. x))) in
    Array1.unsafe_set buf_out i (x *. sigm);
  done;
  out

let relu (a : tensor) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "relu: a not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "relu: out not f32" in
  for i = 0 to n - 1 do
    let x = Array1.unsafe_get buf_a i in
    Array1.unsafe_set buf_out i (if x > 0.0 then x else 0.0);
  done;
  out

let sigmoid (a : tensor) : tensor =
  let n = Tensor.nelements a in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy a.ne) a.n_dims in
  let buf_a = match a.data_f32 with Some buf -> buf | None -> failwith "sigmoid: a not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "sigmoid: out not f32" in
  for i = 0 to n - 1 do
    let x = Array1.unsafe_get buf_a i in
    Array1.unsafe_set buf_out i (1.0 /. (1.0 +. exp (-. x)));
  done;
  out

let swiglu (gate : tensor) (up : tensor) : tensor =
  let n = Tensor.nelements gate in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy gate.ne) gate.n_dims in
  let buf_g = match gate.data_f32 with Some buf -> buf | None -> failwith "swiglu: gate not f32" in
  let buf_u = match up.data_f32 with Some buf -> buf | None -> failwith "swiglu: up not f32" in
  let buf_out = match out.data_f32 with Some buf -> buf | None -> failwith "swiglu: out not f32" in
  for i = 0 to n - 1 do
    let g = Array1.unsafe_get buf_g i in
    let u = Array1.unsafe_get buf_u i in
    let silu_g = g /. (1.0 +. exp (-. g)) in
    Array1.unsafe_set buf_out i (silu_g *. u);
  done;
  out

(* RMS Norm: y_i = (x_i / sqrt(mean(x^2) + eps)) * w_i *)
let rms_norm (x : tensor) (w : tensor) (eps : float) : tensor =
  let d0 = x.ne.(0) in
  let total_rows = Tensor.nelements x / d0 in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy x.ne) x.n_dims in
  let buf_x = match x.data_f32 with Some b -> b | None -> failwith "rms_norm: x not f32" in
  let buf_w = match w.data_f32 with
    | Some b -> b
    | None ->
      match w.data_raw with
      | Some raw ->
        let temp = Array1.create float32 c_layout d0 in
        (match w.qtype with
         | TYPE_F16 -> Quant.dequantize_row_f16 raw 0 temp 0 d0
         | TYPE_Q4_0 -> Quant.dequantize_row_q4_0 raw 0 temp 0 d0
         | TYPE_Q8_0 -> Quant.dequantize_row_q8_0 raw 0 temp 0 d0
         | _ -> failwith "Unsupported weight format in rms_norm");
        temp
      | None -> failwith "rms_norm: w has no data"
  in
  let buf_out = match out.data_f32 with Some b -> b | None -> failwith "rms_norm: out not f32" in

  for row = 0 to total_rows - 1 do
    let off = row * d0 in
    let sum_sq = ref 0.0 in
    for i = 0 to d0 - 1 do
      let v = Array1.unsafe_get buf_x (off + i) in
      sum_sq := !sum_sq +. (v *. v);
    done;
    let mean_sq = !sum_sq /. float_of_int d0 in
    let scale = 1.0 /. sqrt (mean_sq +. eps) in
    for i = 0 to d0 - 1 do
      let v = Array1.unsafe_get buf_x (off + i) in
      let w_val = Array1.unsafe_get buf_w i in
      Array1.unsafe_set buf_out (off + i) (v *. scale *. w_val);
    done;
  done;
  out

(* Layer Norm: y_i = ((x_i - mean) / sqrt(var + eps)) * w_i + b_i *)
let layer_norm (x : tensor) (w : tensor) (b_opt : tensor option) (eps : float) : tensor =
  let d0 = x.ne.(0) in
  let total_rows = Tensor.nelements x / d0 in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy x.ne) x.n_dims in
  let buf_x = match x.data_f32 with Some b -> b | None -> failwith "layer_norm: x not f32" in
  let buf_w = match w.data_f32 with Some b -> b | None -> failwith "layer_norm: w not f32" in
  let buf_b = match b_opt with
    | Some b -> (match b.data_f32 with Some buf -> Some buf | None -> None)
    | None -> None
  in
  let buf_out = match out.data_f32 with Some b -> b | None -> failwith "layer_norm: out not f32" in

  for row = 0 to total_rows - 1 do
    let off = row * d0 in
    let sum = ref 0.0 in
    for i = 0 to d0 - 1 do
      sum := !sum +. Array1.unsafe_get buf_x (off + i);
    done;
    let mean = !sum /. float_of_int d0 in
    let var_sum = ref 0.0 in
    for i = 0 to d0 - 1 do
      let diff = Array1.unsafe_get buf_x (off + i) -. mean in
      var_sum := !var_sum +. (diff *. diff);
    done;
    let variance = !var_sum /. float_of_int d0 in
    let inv_std = 1.0 /. sqrt (variance +. eps) in
    for i = 0 to d0 - 1 do
      let diff = Array1.unsafe_get buf_x (off + i) -. mean in
      let w_val = Array1.unsafe_get buf_w i in
      let b_val = match buf_b with Some bb -> Array1.unsafe_get bb i | None -> 0.0 in
      Array1.unsafe_set buf_out (off + i) (diff *. inv_std *. w_val +. b_val);
    done;
  done;
  out

(* Numerically stable softmax *)
let soft_max (x : tensor) : tensor =
  let d0 = x.ne.(0) in
  let total_rows = Tensor.nelements x / d0 in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy x.ne) x.n_dims in
  let buf_x = match x.data_f32 with Some b -> b | None -> failwith "softmax: x not f32" in
  let buf_out = match out.data_f32 with Some b -> b | None -> failwith "softmax: out not f32" in

  for row = 0 to total_rows - 1 do
    let off = row * d0 in
    let max_val = ref (-. Float.infinity) in
    for i = 0 to d0 - 1 do
      let v = Array1.unsafe_get buf_x (off + i) in
      if v > !max_val then max_val := v;
    done;
    let sum_exp = ref 0.0 in
    for i = 0 to d0 - 1 do
      let v = Array1.unsafe_get buf_x (off + i) in
      let exp_v = exp (v -. !max_val) in
      Array1.unsafe_set buf_out (off + i) exp_v;
      sum_exp := !sum_exp +. exp_v;
    done;
    let inv_sum = 1.0 /. !sum_exp in
    for i = 0 to d0 - 1 do
      let exp_v = Array1.unsafe_get buf_out (off + i) in
      Array1.unsafe_set buf_out (off + i) (exp_v *. inv_sum);
    done;
  done;
  out

let soft_max_ext (x : tensor) (scale_factor : float) (is_causal : bool) : tensor =
  let d0 = x.ne.(0) in
  let total_rows = Tensor.nelements x / d0 in
  let out = Tensor.create_tensor TYPE_F32 (Array.copy x.ne) x.n_dims in
  let buf_x = match x.data_f32 with Some b -> b | None -> failwith "softmax_ext: x not f32" in
  let buf_out = match out.data_f32 with Some b -> b | None -> failwith "softmax_ext: out not f32" in

  for row = 0 to total_rows - 1 do
    let off = row * d0 in
    let max_val = ref (-. Float.infinity) in
    for i = 0 to d0 - 1 do
      if not is_causal || i <= row then begin
        let v = Array1.unsafe_get buf_x (off + i) *. scale_factor in
        if v > !max_val then max_val := v;
      end
    done;
    let sum_exp = ref 0.0 in
    for i = 0 to d0 - 1 do
      if is_causal && i > row then begin
        Array1.unsafe_set buf_out (off + i) 0.0;
      end else begin
        let v = Array1.unsafe_get buf_x (off + i) *. scale_factor in
        let exp_v = exp (v -. !max_val) in
        Array1.unsafe_set buf_out (off + i) exp_v;
        sum_exp := !sum_exp +. exp_v;
      end
    done;
    let inv_sum = if !sum_exp > 0.0 then 1.0 /. !sum_exp else 0.0 in
    for i = 0 to d0 - 1 do
      if not is_causal || i <= row then begin
        let exp_v = Array1.unsafe_get buf_out (off + i) in
        Array1.unsafe_set buf_out (off + i) (exp_v *. inv_sum);
      end
    done;
  done;
  out

(* Rotary Position Embedding (RoPE) *)
let rope (x : tensor) (n_past : int) (n_dims : int) (mode : int) (freq_base : float) (freq_scale : float) (_ext_factor : int) : tensor =
  let ne0 = x.ne.(0) in (* head_dim *)
  let ne1 = x.ne.(1) in (* n_tokens *)
  let ne2 = x.ne.(2) in (* n_heads *)
  let ne3 = x.ne.(3) in (* batch_size *)
  let out = Tensor.create_tensor TYPE_F32 (Array.copy x.ne) x.n_dims in
  let buf_x = match x.data_f32 with Some b -> b | None -> failwith "rope: x not f32" in
  let buf_out = match out.data_f32 with Some b -> b | None -> failwith "rope: out not f32" in

  for b = 0 to ne3 - 1 do
    for h = 0 to ne2 - 1 do
      for t = 0 to ne1 - 1 do
        let pos = n_past + t in
        let row_off = ((b * ne2 + h) * ne1 + t) * ne0 in
        for k = 0 to (n_dims / 2) - 1 do
          let theta = (float_of_int pos /. (freq_base ** (float_of_int (2 * k) /. float_of_int n_dims))) *. freq_scale in
          let cos_th = cos theta in
          let sin_th = sin theta in
          if mode = 0 then begin
            (* LLaMA style (adjacent pairs: 2k, 2k+1) *)
            let x0 = Array1.unsafe_get buf_x (row_off + 2 * k) in
            let x1 = Array1.unsafe_get buf_x (row_off + 2 * k + 1) in
            Array1.unsafe_set buf_out (row_off + 2 * k) (x0 *. cos_th -. x1 *. sin_th);
            Array1.unsafe_set buf_out (row_off + 2 * k + 1) (x0 *. sin_th +. x1 *. cos_th);
          end else begin
            (* NeOX style (split halves: k, k + n_dims/2) *)
            let half = n_dims / 2 in
            let x0 = Array1.unsafe_get buf_x (row_off + k) in
            let x1 = Array1.unsafe_get buf_x (row_off + k + half) in
            Array1.unsafe_set buf_out (row_off + k) (x0 *. cos_th -. x1 *. sin_th);
            Array1.unsafe_set buf_out (row_off + k + half) (x0 *. sin_th +. x1 *. cos_th);
          end
        done;
        (* Copy unchanged high dimensions if n_dims < ne0 *)
        for k = n_dims to ne0 - 1 do
          Array1.unsafe_set buf_out (row_off + k) (Array1.unsafe_get buf_x (row_off + k));
        done;
      done;
    done;
  done;
  out

(* General Matrix Multiplication (GEMM / mul_mat): C = A * B *)
(* A is [K, M] (weights), B is [K, N] (activations/tokens) -> C is [M, N] *)
let mul_mat (w : tensor) (a : tensor) : tensor =
  let k = w.ne.(0) in
  let m = w.ne.(1) in
  let n = a.ne.(1) in
  let out = Tensor.create_2d TYPE_F32 m n in
  let buf_out = match out.data_f32 with Some b -> b | None -> failwith "mul_mat: out not f32" in
  let buf_a = match a.data_f32 with Some b -> b | None -> failwith "mul_mat: a not f32" in

  let num_threads = min 8 (max 1 (try int_of_string (Sys.getenv "LLAMAML_THREADS") with _ -> 4)) in
  let chunk_size = max 32 ((m + num_threads - 1) / num_threads) in

  let compute_range r_start r_end =
    match w.qtype with
    | TYPE_F32 ->
      let buf_w = match w.data_f32 with Some b -> b | None -> failwith "mul_mat: w not f32" in
      for col = 0 to n - 1 do
        let a_off = col * k in
        for row = r_start to r_end - 1 do
          let w_off = row * k in
          let dot = Quant.vec_dot_f32_f32 buf_w w_off buf_a a_off k in
          Array1.unsafe_set buf_out (col * m + row) dot;
        done;
      done

    | TYPE_Q4_0 ->
      let raw_w = match w.data_raw with Some b -> b | None -> failwith "mul_mat: w not raw" in
      let q8_a = Array1.create int8_unsigned c_layout ((k / 32) * 34) in
      let row_bytes = (k / 32) * 18 in
      for col = 0 to n - 1 do
        Quant.quantize_row_q8_0 buf_a (col * k) q8_a 0 k;
        for row = r_start to r_end - 1 do
          let w_off = row * row_bytes in
          let dot = Quant.vec_dot_q4_0_q8_0 raw_w w_off q8_a 0 k in
          Array1.unsafe_set buf_out (col * m + row) dot;
        done;
      done

    | TYPE_Q8_0 ->
      let raw_w = match w.data_raw with Some b -> b | None -> failwith "mul_mat: w not raw" in
      let q8_a = Array1.create int8_unsigned c_layout ((k / 32) * 34) in
      let row_bytes = (k / 32) * 34 in
      for col = 0 to n - 1 do
        Quant.quantize_row_q8_0 buf_a (col * k) q8_a 0 k;
        for row = r_start to r_end - 1 do
          let w_off = row * row_bytes in
          let dot = Quant.vec_dot_q8_0_q8_0 raw_w w_off q8_a 0 k in
          Array1.unsafe_set buf_out (col * m + row) dot;
        done;
      done

    | TYPE_Q4_K ->
      let raw_w = match w.data_raw with Some b -> b | None -> failwith "mul_mat: w not raw" in
      let q8_a = Array1.create int8_unsigned c_layout ((k / 32) * 34) in
      let row_bytes = (k / 256) * 144 in
      for col = 0 to n - 1 do
        Quant.quantize_row_q8_0 buf_a (col * k) q8_a 0 k;
        for row = r_start to r_end - 1 do
          let w_off = row * row_bytes in
          let dot = Quant.vec_dot_q4_k_q8_k raw_w w_off q8_a 0 k in
          Array1.unsafe_set buf_out (col * m + row) dot;
        done;
      done

    | TYPE_Q6_K ->
      let raw_w = match w.data_raw with Some b -> b | None -> failwith "mul_mat: w not raw" in
      let q8_a = Array1.create int8_unsigned c_layout ((k / 32) * 34) in
      let row_bytes = (k / 256) * 210 in
      for col = 0 to n - 1 do
        Quant.quantize_row_q8_0 buf_a (col * k) q8_a 0 k;
        for row = r_start to r_end - 1 do
          let w_off = row * row_bytes in
          let dot = Quant.vec_dot_q6_k_q8_k raw_w w_off q8_a 0 k in
          Array1.unsafe_set buf_out (col * m + row) dot;
        done;
      done

    | _ ->
      let raw_w = match w.data_raw with Some b -> b | None -> failwith "mul_mat: w raw missing" in
      let temp_w = Array1.create float32 c_layout k in
      let row_bytes = Tensor.nbytes w / m in
      for col = 0 to n - 1 do
        let a_off = col * k in
        for row = r_start to r_end - 1 do
          let w_off = row * row_bytes in
          (match w.qtype with
           | TYPE_F16 -> Quant.dequantize_row_f16 raw_w w_off temp_w 0 k
           | TYPE_Q4_1 -> Quant.dequantize_row_q4_1 raw_w w_off temp_w 0 k
           | TYPE_Q5_K -> Quant.dequantize_row_q5_k raw_w w_off temp_w 0 k
           | TYPE_Q6_K -> Quant.dequantize_row_q6_k raw_w w_off temp_w 0 k
           | _ -> failwith "Unsupported quantization type in mul_mat");
          let dot = Quant.vec_dot_f32_f32 temp_w 0 buf_a a_off k in
          Array1.unsafe_set buf_out (col * m + row) dot;
        done;
      done
  in

  let rec run_chunks r =
    if r < m then begin
      let r_next = min m (r + chunk_size) in
      compute_range r r_next;
      run_chunks r_next;
    end
  in
  run_chunks 0;
  out

(* FlashAttention-2 (Online Tiled Softmax Algorithm) *)
(* Q: [head_dim, n_q, n_head, batch] *)
(* K: [head_dim, n_kv, n_head_kv, batch] *)
(* V: [head_dim, n_kv, n_head_kv, batch] *)
let flash_attention_2 (q : tensor) (k : tensor) (v : tensor) (scale : float) (is_causal : bool) : tensor =
  let head_dim = q.ne.(0) in
  let n_q = q.ne.(1) in
  let n_head = q.ne.(2) in
  let n_kv = k.ne.(1) in
  let n_head_kv = k.ne.(2) in
  let gqa_ratio = n_head / n_head_kv in

  let out = Tensor.create_3d TYPE_F32 head_dim n_q n_head in
  let buf_q = match q.data_f32 with Some b -> b | None -> failwith "flash_attn: q not f32" in
  let buf_k = match k.data_f32 with Some b -> b | None -> failwith "flash_attn: k not f32" in
  let buf_v = match v.data_f32 with Some b -> b | None -> failwith "flash_attn: v not f32" in
  let buf_out = match out.data_f32 with Some b -> b | None -> failwith "flash_attn: out not f32" in

  for h = 0 to n_head - 1 do
    let h_kv = h / gqa_ratio in
    for i = 0 to n_q - 1 do
      let q_off = (h * n_q + i) * head_dim in
      let out_off = (h * n_q + i) * head_dim in

      (* Online Softmax accumulators *)
      let m_prev = ref (-. Float.infinity) in
      let l_prev = ref 0.0 in
      let acc = Array.make head_dim 0.0 in

      let max_j = if is_causal then min (i + (n_kv - n_q)) (n_kv - 1) else n_kv - 1 in
      for j = 0 to max_j do
        let k_off = (h_kv * n_kv + j) * head_dim in
        let v_off = (h_kv * n_kv + j) * head_dim in

        (* S_ij = (Q_i . K_j) * scale *)
        let s_ij = Quant.vec_dot_f32_f32 buf_q q_off buf_k k_off head_dim *. scale in
        let m_curr = max !m_prev s_ij in
        let exp_diff = exp (!m_prev -. m_curr) in
        let exp_s = exp (s_ij -. m_curr) in

        let l_curr = !l_prev *. exp_diff +. exp_s in

        (* Update output accumulator *)
        for d = 0 to head_dim - 1 do
          acc.(d) <- acc.(d) *. exp_diff +. exp_s *. Array1.unsafe_get buf_v (v_off + d);
        done;

        m_prev := m_curr;
        l_prev := l_curr;
      done;

      let inv_l = if !l_prev > 0.0 then 1.0 /. !l_prev else 0.0 in
      for d = 0 to head_dim - 1 do
        Array1.unsafe_set buf_out (out_off + d) (acc.(d) *. inv_l);
      done;
    done;
  done;
  out

(* Embedding row lookup: token IDs -> [n_embd, n_tokens] *)
let get_rows (embd : tensor) (token_ids : int array) : tensor =
  let n_embd = embd.ne.(0) in
  let n_tokens = Array.length token_ids in
  let out = Tensor.create_2d TYPE_F32 n_embd n_tokens in
  let buf_out = match out.data_f32 with Some b -> b | None -> failwith "get_rows: out not f32" in

  match embd.qtype with
  | TYPE_F32 ->
    let buf_embd = match embd.data_f32 with Some b -> b | None -> failwith "get_rows: embd not f32" in
    for t = 0 to n_tokens - 1 do
      let tid = token_ids.(t) in
      let src_off = tid * n_embd in
      let dst_off = t * n_embd in
      for i = 0 to n_embd - 1 do
        Array1.unsafe_set buf_out (dst_off + i) (Array1.unsafe_get buf_embd (src_off + i));
      done;
    done;
    out

  | _ ->
    let raw_embd = match embd.data_raw with Some b -> b | None -> failwith "get_rows: raw embd missing" in
    let row_bytes = Tensor.nbytes embd / embd.ne.(1) in
    for t = 0 to n_tokens - 1 do
      let tid = token_ids.(t) in
      let src_off = tid * row_bytes in
      let dst_off = t * n_embd in
      (match embd.qtype with
       | TYPE_F16 -> Quant.dequantize_row_f16 raw_embd src_off buf_out dst_off n_embd
       | TYPE_Q4_0 -> Quant.dequantize_row_q4_0 raw_embd src_off buf_out dst_off n_embd
       | TYPE_Q8_0 -> Quant.dequantize_row_q8_0 raw_embd src_off buf_out dst_off n_embd
       | TYPE_Q4_K -> Quant.dequantize_row_q4_k raw_embd src_off buf_out dst_off n_embd
       | TYPE_Q5_K -> Quant.dequantize_row_q5_k raw_embd src_off buf_out dst_off n_embd
       | TYPE_Q6_K -> Quant.dequantize_row_q6_k raw_embd src_off buf_out dst_off n_embd
       | _ -> failwith "Unsupported embedding quant type in get_rows");
    done;
  out

(* Mixture of Experts top-k gating *)
let moe_gate (logits : tensor) (k : int) : (int array * float array) array =
  let n_exp = logits.ne.(0) in
  let n_tokens = logits.ne.(1) in
  let buf_l = match logits.data_f32 with Some b -> b | None -> failwith "moe_gate: not f32" in
  let results = Array.make n_tokens ([||], [||]) in

  for t = 0 to n_tokens - 1 do
    let off = t * n_exp in
    let pairs = Array.init n_exp (fun i -> (i, Array1.unsafe_get buf_l (off + i))) in
    Array.sort (fun (_, s1) (_, s2) -> Float.compare s2 s1) pairs;

    let top_indices = Array.make k 0 in
    let top_scores = Array.make k 0.0 in
    let max_s = snd pairs.(0) in
    let sum_exp = ref 0.0 in

    for i = 0 to k - 1 do
      let idx, s = pairs.(i) in
      top_indices.(i) <- idx;
      let exp_s = exp (s -. max_s) in
      top_scores.(i) <- exp_s;
      sum_exp := !sum_exp +. exp_s;
    done;

    let inv_sum = if !sum_exp > 0.0 then 1.0 /. !sum_exp else 0.0 in
    for i = 0 to k - 1 do
      top_scores.(i) <- top_scores.(i) *. inv_sum;
    done;

    results.(t) <- (top_indices, top_scores);
  done;
  results
