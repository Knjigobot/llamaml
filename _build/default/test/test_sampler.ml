(* test_sampler.ml - Sampling Pipeline Tests *)

open Llamaml
open Llamaml.Types

let test_greedy_sample () =
  let logits = [| 0.1; 0.5; 0.9; 0.2 |] in
  let chosen = Sampler.sample_greedy logits in
  Alcotest.(check int) "greedy picks index 2" 2 chosen

let test_repetition_penalty () =
  let logits = [| 1.0; 2.0; 3.0 |] in
  Sampler.apply_repetition_penalties logits [2] 2.0 0.0 0.0 10;
  (* Index 2 had logit 3.0, with penalty 2.0 it becomes 1.5 *)
  Alcotest.(check bool) "penalty applied" true (logits.(2) < 2.0)

let () =
  let open Alcotest in
  run "Llamaml.Sampler" [
    "sampler", [
      test_case "greedy" `Quick test_greedy_sample;
      test_case "repetition_penalty" `Quick test_repetition_penalty;
    ];
  ]
