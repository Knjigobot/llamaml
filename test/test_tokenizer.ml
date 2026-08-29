(* test_tokenizer.ml - Tokenizer Encode & Decode Tests *)

open Llamaml
open Llamaml.Types

let test_tokenizer_roundtrip () =
  let tok = Tokenizer.create ~t_type:TOKENIZER_BPE () in
  Tokenizer.add_token tok 0 "<unk>" 0.0 2;
  Tokenizer.add_token tok 1 "<s>" 0.0 3;
  Tokenizer.add_token tok 2 "</s>" 0.0 3;
  Tokenizer.add_token tok 3 "H" 0.0 1;
  Tokenizer.add_token tok 4 "e" 0.0 1;
  Tokenizer.add_token tok 5 "l" 0.0 1;
  Tokenizer.add_token tok 6 "o" 0.0 1;
  Tokenizer.add_token tok 7 "Hel" 1.0 1;
  Tokenizer.add_token tok 8 "lo" 1.0 1;
  Tokenizer.add_token tok 9 "Hello" 2.0 1;

  let encoded = Tokenizer.encode tok ~bos:false ~eos:false "Hello" in
  Alcotest.(check int) "encoded 1 token" 1 (Array.length encoded);
  Alcotest.(check int) "encoded id is 9" 9 encoded.(0);

  let decoded = Tokenizer.decode tok encoded in
  Alcotest.(check string) "decoded string" "Hello" decoded

let () =
  let open Alcotest in
  run "Llamaml.Tokenizer" [
    "tokenizer", [
      test_case "roundtrip" `Quick test_tokenizer_roundtrip;
    ];
  ]
