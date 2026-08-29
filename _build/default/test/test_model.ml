(* test_model.ml - Universal Transformer Model Forward Pass Tests *)

open Llamaml
open Llamaml.Types

let test_transformer_forward () =
  let hp = {
    default_hyperparameters with
    n_vocab = 16;
    n_embd = 8;
    n_head = 2;
    n_head_kv = 2;
    n_layer = 1;
    n_ff = 16;
    n_rot = 4;
  } in
  let head_dim = hp.n_embd / hp.n_head in
  let kv_cache = Kv_cache.create ~n_layer:hp.n_layer ~max_seq_len:32 ~n_head_kv:hp.n_head_kv ~head_dim in
  let tokenizer = Tokenizer.create () in
  for i = 0 to 15 do
    Tokenizer.add_token tokenizer i (Printf.sprintf "t%d" i) 0.0 1;
  done;

  let embd = Tensor.from_f32_array [| 8; 16 |] (Array.make 128 0.1) in
  let norm = Tensor.from_f32_array [| 8 |] (Array.make 8 1.0) in
  let wq = Tensor.from_f32_array [| 8; 8 |] (Array.make 64 0.1) in
  let wk = Tensor.from_f32_array [| 8; 8 |] (Array.make 64 0.1) in
  let wv = Tensor.from_f32_array [| 8; 8 |] (Array.make 64 0.1) in
  let wo = Tensor.from_f32_array [| 8; 8 |] (Array.make 64 0.1) in
  let ffn_norm = Tensor.from_f32_array [| 8 |] (Array.make 8 1.0) in
  let w_gate = Tensor.from_f32_array [| 8; 16 |] (Array.make 128 0.1) in
  let w_up = Tensor.from_f32_array [| 8; 16 |] (Array.make 128 0.1) in
  let w_down = Tensor.from_f32_array [| 16; 8 |] (Array.make 128 0.1) in

  let layer0 = {
    attn_norm = Some norm;
    attn_norm_2 = None;
    wq = Some wq;
    wk = Some wk;
    wv = Some wv;
    wo = Some wo;
    ffn_norm = Some ffn_norm;
    ffn_norm_2 = None;
    w_gate = Some w_gate;
    w_up = Some w_up;
    w_down = Some w_down;
    w_gate_exp = [];
    w_up_exp = [];
    w_down_exp = [];
    mla_q_lora = None;
    mla_kv_lora = None;
  } in

  let weights = {
    token_embd = Some embd;
    output_norm = Some norm;
    output_head = Some embd;
    layers = [| layer0 |];
  } in

  let dummy_gguf = {
    version = 3;
    metadata = Hashtbl.create 8;
    tensor_infos = Hashtbl.create 8;
    tensor_list = [];
    data_offset = 0L;
    file_path = "";
    raw_buffer = None;
  } in

  let model : Model.model_instance = {
    hp;
    weights;
    tokenizer;
    kv_cache;
    gguf = dummy_gguf;
  } in

  let logits = Model.forward model ~token_ids:[| 1; 2; 3 |] ~n_past:0 in
  Alcotest.(check int) "logits dimension 0 is n_vocab" 16 logits.ne.(0);
  Alcotest.(check int) "logits dimension 1 is 1" 1 logits.ne.(1)

let () =
  let open Alcotest in
  run "Llamaml.Model" [
    "transformer", [
      test_case "forward_pass" `Quick test_transformer_forward;
    ];
  ]
