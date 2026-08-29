(* test_gguf.ml - GGUF Binary Deserializer Tests *)

open Llamaml
open Llamaml.Types

let test_gguf_synthetic () =
  (* Create a minimal valid GGUF file in memory *)
  let temp_path = Filename.temp_file "test_gguf" ".gguf" in
  let oc = open_out_bin temp_path in

  (* 1. Magic *)
  output_string oc "GGUF";
  (* 2. Version 3 *)
  output_byte oc 3; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  (* 3. Tensor count: 1 *)
  output_byte oc 1; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  (* 4. Metadata count: 1 *)
  output_byte oc 1; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;

  (* Metadata: general.architecture = "llama" *)
  let key = "general.architecture" in
  let key_len = String.length key in
  output_byte oc (key_len land 0xFF); output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_string oc key;
  (* Type: 8 = STRING *)
  output_byte oc 8; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  let val_str = "llama" in
  let val_len = String.length val_str in
  output_byte oc (val_len land 0xFF); output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_string oc val_str;

  (* Tensor Info: "token_embd.weight" *)
  let tname = "token_embd.weight" in
  let tlen = String.length tname in
  output_byte oc (tlen land 0xFF); output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_string oc tname;
  (* n_dims = 2 *)
  output_byte oc 2; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  (* ne0 = 4, ne1 = 2 *)
  output_byte oc 4; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 2; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  (* type = 0 (F32) *)
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  (* offset = 0 *)
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;
  output_byte oc 0; output_byte oc 0; output_byte oc 0; output_byte oc 0;

  (* Pad to 32 bytes *)
  let cur = pos_out oc in
  let rem = cur mod 32 in
  if rem <> 0 then for _ = 1 to 32 - rem do output_byte oc 0 done;

  (* Tensor Data: 8 floats *)
  for i = 1 to 8 do
    let bits = Int32.bits_of_float (float_of_int i) in
    output_byte oc (Int32.to_int (Int32.logand bits 0xFFl));
    output_byte oc (Int32.to_int (Int32.logand (Int32.shift_right_logical bits 8) 0xFFl));
    output_byte oc (Int32.to_int (Int32.logand (Int32.shift_right_logical bits 16) 0xFFl));
    output_byte oc (Int32.to_int (Int32.logand (Int32.shift_right_logical bits 24) 0xFFl));
  done;
  close_out oc;

  let gguf = Gguf.parse_file temp_path in
  Alcotest.(check int) "gguf version" 3 gguf.version;
  let arch = Gguf.get_metadata_string gguf "general.architecture" in
  Alcotest.(check (option string)) "architecture" (Some "llama") arch;
  let t = Gguf.load_tensor gguf "token_embd.weight" in
  Alcotest.(check int) "tensor ne0" 4 t.ne.(0);
  Alcotest.(check int) "tensor ne1" 2 t.ne.(1);
  (try Sys.remove temp_path with _ -> ())

let () =
  let open Alcotest in
  run "Llamaml.Gguf" [
    "gguf_parser", [
      test_case "synthetic_gguf_file" `Quick test_gguf_synthetic;
    ];
  ]
