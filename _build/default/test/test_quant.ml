(* test_quant.ml - Quantization Kernels & Vectorized Math Tests *)

open Llamaml
open Llamaml.Types
open Bigarray

let test_f16_roundtrip () =
  let test_vals = [0.0; 1.0; -1.0; 0.5; -0.5; 2.5; 100.0; -42.0; 0.001] in
  List.iter (fun v ->
    let h = Quant.f32_to_f16 v in
    let decoded = Quant.f16_to_f32 h in
    let diff = abs_float (v -. decoded) in
    Alcotest.(check bool) (Printf.sprintf "f16 roundtrip for %f" v) true (diff < 0.05)
  ) test_vals

let test_q8_0_quant_dequant () =
  let k = 32 in
  let src = Array1.create float32 c_layout k in
  for i = 0 to k - 1 do
    Array1.unsafe_set src i (float_of_int (i - 16) *. 0.5);
  done;
  let q_buf = Array1.create int8_unsigned c_layout 34 in
  Quant.quantize_row_q8_0 src 0 q_buf 0 k;
  let dst = Array1.create float32 c_layout k in
  Quant.dequantize_row_q8_0 q_buf 0 dst 0 k;
  for i = 0 to k - 1 do
    let orig = Array1.unsafe_get src i in
    let deq = Array1.unsafe_get dst i in
    let err = abs_float (orig -. deq) in
    Alcotest.(check bool) "q8_0 error within bound" true (err < 0.05);
  done

let test_q4_0_quant_dequant () =
  let k = 32 in
  let src = Array1.create float32 c_layout k in
  for i = 0 to k - 1 do
    Array1.unsafe_set src i (float_of_int (i mod 8 - 4) *. 0.25);
  done;
  let q_buf = Array1.create int8_unsigned c_layout 18 in
  Quant.quantize_row_q4_0 src 0 q_buf 0 k;
  let dst = Array1.create float32 c_layout k in
  Quant.dequantize_row_q4_0 q_buf 0 dst 0 k;
  for i = 0 to k - 1 do
    let orig = Array1.unsafe_get src i in
    let deq = Array1.unsafe_get dst i in
    let err = abs_float (orig -. deq) in
    Alcotest.(check bool) "q4_0 error within bound" true (err < 0.25);
  done

let test_vector_dot_products () =
  let k = 32 in
  let x = Array1.create float32 c_layout k in
  let y = Array1.create float32 c_layout k in
  for i = 0 to k - 1 do
    Array1.unsafe_set x i 1.0;
    Array1.unsafe_set y i 2.0;
  done;
  let dot_f32 = Quant.vec_dot_f32_f32 x 0 y 0 k in
  Alcotest.(check bool) "f32 dot product" true (abs_float (dot_f32 -. 64.0) < 1e-5);

  let q8_x = Array1.create int8_unsigned c_layout 34 in
  let q8_y = Array1.create int8_unsigned c_layout 34 in
  Quant.quantize_row_q8_0 x 0 q8_x 0 k;
  Quant.quantize_row_q8_0 y 0 q8_y 0 k;
  let dot_q8 = Quant.vec_dot_q8_0_q8_0 q8_x 0 q8_y 0 k in
  Alcotest.(check bool) "q8_0 dot product" true (abs_float (dot_q8 -. 64.0) < 0.5)

let () =
  let open Alcotest in
  run "Llamaml.Quant" [
    "f16", [
      test_case "f16_roundtrip" `Quick test_f16_roundtrip;
    ];
    "quantization", [
      test_case "q8_0_roundtrip" `Quick test_q8_0_quant_dequant;
      test_case "q4_0_roundtrip" `Quick test_q4_0_quant_dequant;
    ];
    "vector_dot", [
      test_case "dot_products" `Quick test_vector_dot_products;
    ];
  ]
