(* quant.ml - Quantization Kernels & Vectorized Mathematical Operations *)

open Types
open Bigarray

(* IEEE 754 Half-Precision Float16 to Float32 conversion *)
let f16_to_f32 (h : int) : float =
  let sign = (h lsr 15) land 1 in
  let exp = (h lsr 10) land 0x1F in
  let mant = h land 0x3FF in
  let s_mult = if sign = 1 then -1.0 else 1.0 in
  if exp = 0 then
    if mant = 0 then s_mult *. 0.0
    else s_mult *. (ldexp (float_of_int mant) (-24))
  else if exp = 31 then
    if mant = 0 then s_mult *. Float.infinity
    else Float.nan
  else
    let norm_mant = 1.0 +. (float_of_int mant /. 1024.0) in
    s_mult *. (ldexp norm_mant (exp - 15))

(* Float32 to IEEE 754 Half-Precision Float16 conversion *)
let f32_to_f16 (f : float) : int =
  if Float.is_nan f then 0x7E00
  else if f = Float.infinity then 0x7C00
  else if f = Float.neg_infinity then 0xFC00
  else if f = 0.0 then
    if Float.copy_sign 1.0 f < 0.0 then 0x8000 else 0x0000
  else
    let sign = if f < 0.0 then 1 else 0 in
    let abs_f = abs_float f in
    let mant, exp = frexp abs_f in
    (* abs_f = mant * 2^exp where 0.5 <= mant < 1.0 *)
    let biased_exp = exp + 14 in
    if biased_exp <= 0 then
      (* Subnormal *)
      let m = int_of_float (mant *. (ldexp 1.0 (biased_exp + 10))) in
      (sign lsl 15) lor (m land 0x3FF)
    else if biased_exp >= 31 then
      (sign lsl 15) lor 0x7C00 (* Overflow to inf *)
    else
      let m = int_of_float ((mant *. 2.0 -. 1.0) *. 1024.0 +. 0.5) land 0x3FF in
      (sign lsl 15) lor (biased_exp lsl 10) lor m

let bf16_to_f32 (b : int) : float =
  let bits = Int32.shift_left (Int32.of_int b) 16 in
  Int32.float_of_bits bits

let f32_to_bf16 (f : float) : int =
  let bits = Int32.bits_of_float f in
  Int32.to_int (Int32.shift_right_logical bits 16) land 0xFFFF

let read_u16_le (buf : u8_buffer) (offset : int) : int =
  let b0 = Array1.unsafe_get buf offset in
  let b1 = Array1.unsafe_get buf (offset + 1) in
  b0 lor (b1 lsl 8)

let write_u16_le (buf : u8_buffer) (offset : int) (v : int) : unit =
  Array1.unsafe_set buf offset (v land 0xFF);
  Array1.unsafe_set buf (offset + 1) ((v lsr 8) land 0xFF)

(* Dequantize Q4_0: 32 values per 18-byte block (2 bytes f16 delta + 16 bytes nibbles) *)
let dequantize_row_q4_0 (src : u8_buffer) (src_off : int) (dst : f32_buffer) (dst_off : int) (k : int) : unit =
  let nb = k / 32 in
  let s_off = ref src_off in
  let d_off = ref dst_off in
  for _b = 0 to nb - 1 do
    let d_raw = read_u16_le src !s_off in
    let d = f16_to_f32 d_raw in
    s_off := !s_off + 2;
    for i = 0 to 15 do
      let byte_val = Array1.unsafe_get src (!s_off + i) in
      let x0 = (byte_val land 0x0F) - 8 in
      let x1 = ((byte_val lsr 4) land 0x0F) - 8 in
      Array1.unsafe_set dst (!d_off + i) (float_of_int x0 *. d);
      Array1.unsafe_set dst (!d_off + i + 16) (float_of_int x1 *. d);
    done;
    s_off := !s_off + 16;
    d_off := !d_off + 32;
  done

(* Dequantize Q4_1: 32 values per 20-byte block (2 bytes f16 d + 2 bytes f16 m + 16 bytes nibbles) *)
let dequantize_row_q4_1 (src : u8_buffer) (src_off : int) (dst : f32_buffer) (dst_off : int) (k : int) : unit =
  let nb = k / 32 in
  let s_off = ref src_off in
  let d_off = ref dst_off in
  for _b = 0 to nb - 1 do
    let d_raw = read_u16_le src !s_off in
    let m_raw = read_u16_le src (!s_off + 2) in
    let d = f16_to_f32 d_raw in
    let m = f16_to_f32 m_raw in
    s_off := !s_off + 4;
    for i = 0 to 15 do
      let byte_val = Array1.unsafe_get src (!s_off + i) in
      let x0 = byte_val land 0x0F in
      let x1 = (byte_val lsr 4) land 0x0F in
      Array1.unsafe_set dst (!d_off + i) (float_of_int x0 *. d +. m);
      Array1.unsafe_set dst (!d_off + i + 16) (float_of_int x1 *. d +. m);
    done;
    s_off := !s_off + 16;
    d_off := !d_off + 32;
  done

(* Dequantize Q8_0: 32 values per 34-byte block (2 bytes f16 delta + 32 bytes int8) *)
let dequantize_row_q8_0 (src : u8_buffer) (src_off : int) (dst : f32_buffer) (dst_off : int) (k : int) : unit =
  let nb = k / 32 in
  let s_off = ref src_off in
  let d_off = ref dst_off in
  for _b = 0 to nb - 1 do
    let d_raw = read_u16_le src !s_off in
    let d = f16_to_f32 d_raw in
    s_off := !s_off + 2;
    for i = 0 to 31 do
      let byte_val = Array1.unsafe_get src (!s_off + i) in
      let signed_val = if byte_val >= 128 then byte_val - 256 else byte_val in
      Array1.unsafe_set dst (!d_off + i) (float_of_int signed_val *. d);
    done;
    s_off := !s_off + 32;
    d_off := !d_off + 32;
  done

(* Dequantize Q4_K: 256 values per 144-byte super-block *)
let dequantize_row_q4_k (src : u8_buffer) (src_off : int) (dst : f32_buffer) (dst_off : int) (k : int) : unit =
  let nb = k / 256 in
  let s_off = ref src_off in
  let d_off = ref dst_off in
  for _b = 0 to nb - 1 do
    let d_raw = read_u16_le src !s_off in
    let dmin_raw = read_u16_le src (!s_off + 2) in
    let d = f16_to_f32 d_raw in
    let dmin = f16_to_f32 dmin_raw in
    let scales_off = !s_off + 4 in
    let qs_off = !s_off + 16 in
    
    (* Unpack 6-bit scales and mins *)
    let scales = Array.make 8 0 in
    let mins = Array.make 8 0 in
    for j = 0 to 3 do
      let sc_byte = Array1.unsafe_get src (scales_off + j) in
      let m_byte = Array1.unsafe_get src (scales_off + j + 4) in
      scales.(j * 2) <- sc_byte land 63;
      scales.(j * 2 + 1) <- ((sc_byte lsr 6) land 3) lor ((Array1.unsafe_get src (scales_off + j + 8) land 0x0F) lsl 2);
      mins.(j * 2) <- m_byte land 63;
      mins.(j * 2 + 1) <- ((m_byte lsr 6) land 3) lor (((Array1.unsafe_get src (scales_off + j + 8) lsr 4) land 0x0F) lsl 2);
    done;

    for sb = 0 to 7 do
      let sc = float_of_int scales.(sb) *. d in
      let mn = float_of_int mins.(sb) *. dmin in
      let sb_qs = qs_off + sb * 16 in
      let sb_dst = !d_off + sb * 32 in
      for i = 0 to 15 do
        let byte_val = Array1.unsafe_get src (sb_qs + i) in
        let q0 = byte_val land 0x0F in
        let q1 = (byte_val lsr 4) land 0x0F in
        Array1.unsafe_set dst (sb_dst + i) (float_of_int q0 *. sc -. mn);
        Array1.unsafe_set dst (sb_dst + i + 16) (float_of_int q1 *. sc -. mn);
      done;
    done;
    s_off := !s_off + 144;
    d_off := !d_off + 256;
  done

(* Dequantize Q5_K: 256 values per 176-byte super-block *)
let dequantize_row_q5_k (src : u8_buffer) (src_off : int) (dst : f32_buffer) (dst_off : int) (k : int) : unit =
  let nb = k / 256 in
  let s_off = ref src_off in
  let d_off = ref dst_off in
  for _b = 0 to nb - 1 do
    let d_raw = read_u16_le src !s_off in
    let dmin_raw = read_u16_le src (!s_off + 2) in
    let d = f16_to_f32 d_raw in
    let dmin = f16_to_f32 dmin_raw in
    let scales_off = !s_off + 4 in
    let qh_off = !s_off + 16 in
    let qs_off = !s_off + 48 in
    
    for sb = 0 to 7 do
      let sc_byte = Array1.unsafe_get src (scales_off + (sb / 2)) in
      let sc = float_of_int (if sb mod 2 = 0 then sc_byte land 0x0F else (sc_byte lsr 4) land 0x0F) *. d in
      let mn = dmin in
      let sb_qs = qs_off + sb * 16 in
      let sb_dst = !d_off + sb * 32 in
      for i = 0 to 15 do
        let byte_val = Array1.unsafe_get src (sb_qs + i) in
        let qh_byte = Array1.unsafe_get src (qh_off + (sb * 4) + (i / 4)) in
        let h0 = (qh_byte lsr ((i mod 4) * 2)) land 1 in
        let h1 = (qh_byte lsr ((i mod 4) * 2 + 1)) land 1 in
        let q0 = (byte_val land 0x0F) lor (h0 lsl 4) in
        let q1 = ((byte_val lsr 4) land 0x0F) lor (h1 lsl 4) in
        Array1.unsafe_set dst (sb_dst + i) (float_of_int q0 *. sc -. mn);
        Array1.unsafe_set dst (sb_dst + i + 16) (float_of_int q1 *. sc -. mn);
      done;
    done;
    s_off := !s_off + 176;
    d_off := !d_off + 256;
  done

(* Dequantize Q6_K: 256 values per 210-byte super-block *)
let dequantize_row_q6_k (src : u8_buffer) (src_off : int) (dst : f32_buffer) (dst_off : int) (k : int) : unit =
  let nb = k / 256 in
  let s_off = ref src_off in
  let d_off = ref dst_off in
  for _b = 0 to nb - 1 do
    let ql_off = !s_off in
    let qh_off = !s_off + 128 in
    let scales_off = !s_off + 192 in
    let d_raw = read_u16_le src (!s_off + 208) in
    let d = f16_to_f32 d_raw in

    for sb = 0 to 15 do
      let sc_byte = Array1.unsafe_get src (scales_off + sb) in
      let sc_signed = if sc_byte >= 128 then sc_byte - 256 else sc_byte in
      let sc = float_of_int sc_signed *. d in
      let sb_dst = !d_off + sb * 16 in
      for i = 0 to 7 do
        let ql_byte = Array1.unsafe_get src (ql_off + sb * 8 + i) in
        let qh_byte = Array1.unsafe_get src (qh_off + sb * 4 + (i / 2)) in
        let h0 = (qh_byte lsr ((i mod 2) * 4)) land 3 in
        let h1 = (qh_byte lsr ((i mod 2) * 4 + 2)) land 3 in
        let q0 = ((ql_byte land 0x0F) lor (h0 lsl 4)) - 32 in
        let q1 = (((ql_byte lsr 4) land 0x0F) lor (h1 lsl 4)) - 32 in
        Array1.unsafe_set dst (sb_dst + i) (float_of_int q0 *. sc);
        Array1.unsafe_set dst (sb_dst + i + 8) (float_of_int q1 *. sc);
      done;
    done;
    s_off := !s_off + 210;
    d_off := !d_off + 256;
  done

(* Dequantize F16: raw bytes to float32 *)
let dequantize_row_f16 (src : u8_buffer) (src_off : int) (dst : f32_buffer) (dst_off : int) (k : int) : unit =
  for i = 0 to k - 1 do
    let raw = read_u16_le src (src_off + i * 2) in
    Array1.unsafe_set dst (dst_off + i) (f16_to_f32 raw);
  done

(* Quantize row to Q8_0 *)
let quantize_row_q8_0 (src : f32_buffer) (src_off : int) (dst : u8_buffer) (dst_off : int) (k : int) : unit =
  let nb = k / 32 in
  let s_off = ref src_off in
  let d_off = ref dst_off in
  for _b = 0 to nb - 1 do
    let amax = ref 0.0 in
    for i = 0 to 31 do
      let v = abs_float (Array1.unsafe_get src (!s_off + i)) in
      if v > !amax then amax := v;
    done;
    let d = !amax /. 127.0 in
    let id = if d > 0.0 then 1.0 /. d else 0.0 in
    let d_f16 = f32_to_f16 d in
    write_u16_le dst !d_off d_f16;
    d_off := !d_off + 2;
    for i = 0 to 31 do
      let v = Array1.unsafe_get src (!s_off + i) in
      let q = int_of_float (Float.round (v *. id)) in
      let clamped = if q < -128 then -128 else if q > 127 then 127 else q in
      let u8 = if clamped < 0 then clamped + 256 else clamped in
      Array1.unsafe_set dst (!d_off + i) u8;
    done;
    s_off := !s_off + 32;
    d_off := !d_off + 32;
  done

(* Quantize row to Q4_0 *)
let quantize_row_q4_0 (src : f32_buffer) (src_off : int) (dst : u8_buffer) (dst_off : int) (k : int) : unit =
  let nb = k / 32 in
  let s_off = ref src_off in
  let d_off = ref dst_off in
  for _b = 0 to nb - 1 do
    let amax = ref 0.0 in
    for i = 0 to 31 do
      let v = abs_float (Array1.unsafe_get src (!s_off + i)) in
      if v > !amax then amax := v;
    done;
    let d = !amax /. -8.0 in
    let id = if d <> 0.0 then 1.0 /. d else 0.0 in
    let d_f16 = f32_to_f16 (-. d) in
    write_u16_le dst !d_off d_f16;
    d_off := !d_off + 2;
    for i = 0 to 15 do
      let v0 = Array1.unsafe_get src (!s_off + i) in
      let v1 = Array1.unsafe_get src (!s_off + i + 16) in
      let x0 = int_of_float (Float.round (v0 *. id +. 8.5)) in
      let x1 = int_of_float (Float.round (v1 *. id +. 8.5)) in
      let c0 = if x0 < 0 then 0 else if x0 > 15 then 15 else x0 in
      let c1 = if x1 < 0 then 0 else if x1 > 15 then 15 else x1 in
      Array1.unsafe_set dst (!d_off + i) (c0 lor (c1 lsl 4));
    done;
    s_off := !s_off + 32;
    d_off := !d_off + 16;
  done

(* High-performance 8-way unrolled vectorized dot products *)

let[@inline always] vec_dot_f32_f32 (x : f32_buffer) (x_off : int) (y : f32_buffer) (y_off : int) (n : int) : float =
  let sum0 = ref 0.0 in
  let sum1 = ref 0.0 in
  let sum2 = ref 0.0 in
  let sum3 = ref 0.0 in
  let n8 = n land (lnot 7) in
  let i = ref 0 in
  while !i < n8 do
    let ix = x_off + !i in
    let iy = y_off + !i in
    sum0 := !sum0 +. (Array1.unsafe_get x ix *. Array1.unsafe_get y iy)
                  +. (Array1.unsafe_get x (ix + 1) *. Array1.unsafe_get y (iy + 1));
    sum1 := !sum1 +. (Array1.unsafe_get x (ix + 2) *. Array1.unsafe_get y (iy + 2))
                  +. (Array1.unsafe_get x (ix + 3) *. Array1.unsafe_get y (iy + 3));
    sum2 := !sum2 +. (Array1.unsafe_get x (ix + 4) *. Array1.unsafe_get y (iy + 4))
                  +. (Array1.unsafe_get x (ix + 5) *. Array1.unsafe_get y (iy + 5));
    sum3 := !sum3 +. (Array1.unsafe_get x (ix + 6) *. Array1.unsafe_get y (iy + 6))
                  +. (Array1.unsafe_get x (ix + 7) *. Array1.unsafe_get y (iy + 7));
    i := !i + 8;
  done;
  let total = ref (!sum0 +. !sum1 +. !sum2 +. !sum3) in
  while !i < n do
    total := !total +. (Array1.unsafe_get x (x_off + !i) *. Array1.unsafe_get y (y_off + !i));
    incr i;
  done;
  !total

let[@inline always] vec_dot_q4_0_q8_0 (w : u8_buffer) (w_off : int) (a : u8_buffer) (a_off : int) (k : int) : float =
  let nb = k / 32 in
  let sum = ref 0.0 in
  let w_ptr = ref w_off in
  let a_ptr = ref a_off in
  for _b = 0 to nb - 1 do
    let d_w = f16_to_f32 (read_u16_le w !w_ptr) in
    let d_a = f16_to_f32 (read_u16_le a !a_ptr) in
    let d = d_w *. d_a in
    w_ptr := !w_ptr + 2;
    a_ptr := !a_ptr + 2;
    
    (* 8-way unrolled accumulator for 32 weights per block *)
    let isum0 = ref 0 in
    let isum1 = ref 0 in
    
    for i = 0 to 7 do
      let byte_w0 = Array1.unsafe_get w (!w_ptr + i) in
      let byte_w1 = Array1.unsafe_get w (!w_ptr + i + 8) in
      
      let q0_0 = (byte_w0 land 0x0F) - 8 in
      let q0_1 = ((byte_w0 lsr 4) land 0x0F) - 8 in
      let q1_0 = (byte_w1 land 0x0F) - 8 in
      let q1_1 = ((byte_w1 lsr 4) land 0x0F) - 8 in
      
      let sa0_0 = let b = Array1.unsafe_get a (!a_ptr + i) in if b >= 128 then b - 256 else b in
      let sa0_1 = let b = Array1.unsafe_get a (!a_ptr + i + 16) in if b >= 128 then b - 256 else b in
      let sa1_0 = let b = Array1.unsafe_get a (!a_ptr + i + 8) in if b >= 128 then b - 256 else b in
      let sa1_1 = let b = Array1.unsafe_get a (!a_ptr + i + 24) in if b >= 128 then b - 256 else b in
      
      isum0 := !isum0 + (q0_0 * sa0_0) + (q0_1 * sa0_1);
      isum1 := !isum1 + (q1_0 * sa1_0) + (q1_1 * sa1_1);
    done;
    sum := !sum +. (float_of_int (!isum0 + !isum1) *. d);
    w_ptr := !w_ptr + 16;
    a_ptr := !a_ptr + 32;
  done;
  !sum

let[@inline always] vec_dot_q8_0_q8_0 (w : u8_buffer) (w_off : int) (a : u8_buffer) (a_off : int) (k : int) : float =
  let nb = k / 32 in
  let sum = ref 0.0 in
  let w_ptr = ref w_off in
  let a_ptr = ref a_off in
  for _b = 0 to nb - 1 do
    let d_w = f16_to_f32 (read_u16_le w !w_ptr) in
    let d_a = f16_to_f32 (read_u16_le a !a_ptr) in
    let d = d_w *. d_a in
    w_ptr := !w_ptr + 2;
    a_ptr := !a_ptr + 2;
    
    let isum0 = ref 0 in
    let isum1 = ref 0 in
    let isum2 = ref 0 in
    let isum3 = ref 0 in
    
    for i = 0 to 7 do
      let sw0 = let b = Array1.unsafe_get w (!w_ptr + i) in if b >= 128 then b - 256 else b in
      let sa0 = let b = Array1.unsafe_get a (!a_ptr + i) in if b >= 128 then b - 256 else b in
      let sw1 = let b = Array1.unsafe_get w (!w_ptr + i + 8) in if b >= 128 then b - 256 else b in
      let sa1 = let b = Array1.unsafe_get a (!a_ptr + i + 8) in if b >= 128 then b - 256 else b in
      let sw2 = let b = Array1.unsafe_get w (!w_ptr + i + 16) in if b >= 128 then b - 256 else b in
      let sa2 = let b = Array1.unsafe_get a (!a_ptr + i + 16) in if b >= 128 then b - 256 else b in
      let sw3 = let b = Array1.unsafe_get w (!w_ptr + i + 24) in if b >= 128 then b - 256 else b in
      let sa3 = let b = Array1.unsafe_get a (!a_ptr + i + 24) in if b >= 128 then b - 256 else b in
      isum0 := !isum0 + (sw0 * sa0);
      isum1 := !isum1 + (sw1 * sa1);
      isum2 := !isum2 + (sw2 * sa2);
      isum3 := !isum3 + (sw3 * sa3);
    done;
    sum := !sum +. (float_of_int (!isum0 + !isum1 + !isum2 + !isum3) *. d);
    w_ptr := !w_ptr + 32;
    a_ptr := !a_ptr + 32;
  done;
  !sum

let static_temp_w = Array1.create float32 c_layout 16384
let static_temp_a = Array1.create float32 c_layout 16384

let[@inline always] vec_dot_q4_k_q8_k (w : u8_buffer) (w_off : int) (a : u8_buffer) (a_off : int) (k : int) : float =
  let nb = k / 256 in
  let sum = ref 0.0 in
  for b = 0 to nb - 1 do
    dequantize_row_q4_k w (w_off + b * 144) static_temp_w 0 256;
    dequantize_row_q8_0 a (a_off + b * (8 * 34)) static_temp_a 0 256;
    sum := !sum +. vec_dot_f32_f32 static_temp_w 0 static_temp_a 0 256;
  done;
  !sum
