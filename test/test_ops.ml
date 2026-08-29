(* test_ops.ml - Mathematical Tensor Operations Tests *)

open Llamaml
open Llamaml.Types

let test_rms_norm () =
  let x = Tensor.from_f32_array [| 4; 1 |] [| 2.0; 2.0; 2.0; 2.0 |] in
  let w = Tensor.from_f32_array [| 4 |] [| 1.0; 1.0; 1.0; 1.0 |] in
  let y = Ops.rms_norm x w 1e-5 in
  let arr = Tensor.to_f32_array y in
  for i = 0 to 3 do
    Alcotest.(check bool) "rms_norm unit output" true (abs_float (arr.(i) -. 1.0) < 1e-4);
  done

let test_silu () =
  let x = Tensor.from_f32_array [| 2 |] [| 0.0; 2.0 |] in
  let y = Ops.silu x in
  let arr = Tensor.to_f32_array y in
  Alcotest.(check bool) "silu(0) = 0" true (abs_float arr.(0) < 1e-5);
  let expected_2 = 2.0 /. (1.0 +. exp (-2.0)) in
  Alcotest.(check bool) "silu(2) exact" true (abs_float (arr.(1) -. expected_2) < 1e-4)

let test_softmax () =
  let x = Tensor.from_f32_array [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let y = Ops.soft_max x in
  let arr = Tensor.to_f32_array y in
  let sum = arr.(0) +. arr.(1) +. arr.(2) in
  Alcotest.(check bool) "softmax sum to 1" true (abs_float (sum -. 1.0) < 1e-5);
  Alcotest.(check bool) "softmax monotonic" true (arr.(2) > arr.(1) && arr.(1) > arr.(0))

let test_mul_mat () =
  (* W: [2, 2], X: [2, 1] *)
  (* W = [[1, 2], [3, 4]], X = [1, 1] -> Y = [3, 7] *)
  let w = Tensor.from_f32_array [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let x = Tensor.from_f32_array [| 2; 1 |] [| 1.0; 1.0 |] in
  let y = Ops.mul_mat w x in
  let arr = Tensor.to_f32_array y in
  Alcotest.(check bool) "mul_mat row 0" true (abs_float (arr.(0) -. 3.0) < 1e-5);
  Alcotest.(check bool) "mul_mat row 1" true (abs_float (arr.(1) -. 7.0) < 1e-5)

let () =
  let open Alcotest in
  run "Llamaml.Ops" [
    "math_operators", [
      test_case "rms_norm" `Quick test_rms_norm;
      test_case "silu" `Quick test_silu;
      test_case "softmax" `Quick test_softmax;
      test_case "mul_mat" `Quick test_mul_mat;
    ];
  ]
