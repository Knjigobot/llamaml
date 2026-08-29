(* quant.mli - Quantization Kernels & Vectorized Mathematical Operations *)

open Types
open Bigarray

val f16_to_f32 : int -> float
val f32_to_f16 : float -> int
val bf16_to_f32 : int -> float
val f32_to_bf16 : float -> int

val dequantize_row_q4_0 : u8_buffer -> int -> f32_buffer -> int -> int -> unit
val dequantize_row_q4_1 : u8_buffer -> int -> f32_buffer -> int -> int -> unit
val dequantize_row_q8_0 : u8_buffer -> int -> f32_buffer -> int -> int -> unit
val dequantize_row_q4_k : u8_buffer -> int -> f32_buffer -> int -> int -> unit
val dequantize_row_q5_k : u8_buffer -> int -> f32_buffer -> int -> int -> unit
val dequantize_row_q6_k : u8_buffer -> int -> f32_buffer -> int -> int -> unit
val dequantize_row_f16 : u8_buffer -> int -> f32_buffer -> int -> int -> unit

val quantize_row_q8_0 : f32_buffer -> int -> u8_buffer -> int -> int -> unit
val quantize_row_q4_0 : f32_buffer -> int -> u8_buffer -> int -> int -> unit

val vec_dot_f32_f32 : f32_buffer -> int -> f32_buffer -> int -> int -> float
val vec_dot_q4_0_q8_0 : u8_buffer -> int -> u8_buffer -> int -> int -> float
val vec_dot_q8_0_q8_0 : u8_buffer -> int -> u8_buffer -> int -> int -> float
val vec_dot_q4_k_q8_k : u8_buffer -> int -> u8_buffer -> int -> int -> float
