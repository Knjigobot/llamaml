(* model.ml - Universal Transformer Forward & Generation Engine *)

open Types
open Bigarray

type model_instance = {
  hp : hyperparameters;
  weights : model_weights;
  tokenizer : tokenizer;
  kv_cache : kv_cache;
  gguf : Gguf.gguf_file;
}

let load (path : string) : model_instance =
  let gguf = Gguf.parse_file path in
  let hp = Gguf.extract_hyperparameters gguf in
  let tokenizer = Gguf.extract_tokenizer gguf in
  let weights = Gguf.load_model_weights gguf hp in
  let head_dim = hp.n_embd / hp.n_head in
  let kv_cache = Kv_cache.create ~n_layer:hp.n_layer ~max_seq_len:4096 ~n_head_kv:hp.n_head_kv ~head_dim in
  {
    hp;
    weights;
    tokenizer;
    kv_cache;
    gguf;
  }

let forward (m : model_instance) ~(token_ids : token_id array) ~(n_past : int) : tensor =
  let n_tokens = Array.length token_ids in
  let n_embd = m.hp.n_embd in
  let n_head = m.hp.n_head in
  let head_dim = n_embd / n_head in
  let eps = m.hp.rms_norm_eps in

  (* 1. Input embedding lookup *)
  let token_embd = match m.weights.token_embd with
    | Some t -> t
    | None -> failwith "Model missing token_embd tensor"
  in
  let x = ref (Ops.get_rows token_embd token_ids) in

  (* 2. Iterate Transformer Layers *)
  for l = 0 to m.hp.n_layer - 1 do
    let layer = m.weights.layers.(l) in
    let attn_norm = match layer.attn_norm with Some t -> t | None -> failwith "Missing attn_norm" in
    let wq = match layer.wq with Some t -> t | None -> failwith "Missing wq" in
    let wk = match layer.wk with Some t -> t | None -> failwith "Missing wk" in
    let wv = match layer.wv with Some t -> t | None -> failwith "Missing wv" in
    let wo = match layer.wo with Some t -> t | None -> failwith "Missing wo" in

    (* Attention Pre-Norm *)
    let x_norm = Ops.rms_norm !x attn_norm eps in

    (* Q, K, V Projections *)
    let q_raw = Ops.mul_mat wq x_norm in
    let k_raw = Ops.mul_mat wk x_norm in
    let v_raw = Ops.mul_mat wv x_norm in

    (* Reshape Q, K, V for multi-head attention *)
    let q_3d = Tensor.reshape q_raw [| head_dim; n_tokens; n_head; 1 |] in
    let k_3d = Tensor.reshape k_raw [| head_dim; n_tokens; m.hp.n_head_kv; 1 |] in
    let v_3d = Tensor.reshape v_raw [| head_dim; n_tokens; m.hp.n_head_kv; 1 |] in

    (* RoPE *)
    let q_rot = Ops.rope q_3d n_past m.hp.n_rot m.hp.rope_type m.hp.rope_freq_base m.hp.rope_freq_scale 0 in
    let k_rot = Ops.rope k_3d n_past m.hp.n_rot m.hp.rope_type m.hp.rope_freq_base m.hp.rope_freq_scale 0 in

    (* Update KV Cache *)
    Kv_cache.update_kv m.kv_cache ~layer:l ~n_past ~k_new:k_rot ~v_new:v_3d;

    (* Retrieve full sequence history from KV cache *)
    let total_seq_len = n_past + n_tokens in
    let k_all = Kv_cache.get_k_view m.kv_cache ~layer:l ~n_tokens:total_seq_len in
    let v_all = Kv_cache.get_v_view m.kv_cache ~layer:l ~n_tokens:total_seq_len in

    (* FlashAttention-2 *)
    let scale_factor = 1.0 /. sqrt (float_of_int head_dim) in
    let attn_out = Ops.flash_attention_2 q_rot k_all v_all scale_factor true in

    (* Reshape and Output Projection *)
    let attn_2d = Tensor.reshape attn_out [| n_embd; n_tokens; 1; 1 |] in
    let q_proj = Ops.mul_mat wo attn_2d in

    (* Residual connection *)
    x := Ops.add !x q_proj;

    (* FFN *)
    let ffn_norm = match layer.ffn_norm with Some t -> t | None -> failwith "Missing ffn_norm" in
    let x_ffn_norm = Ops.rms_norm !x ffn_norm eps in

    let ffn_out =
      match (layer.w_gate, layer.w_up, layer.w_down) with
      | Some w_gate, Some w_up, Some w_down ->
        let gate = Ops.mul_mat w_gate x_ffn_norm in
        let up = Ops.mul_mat w_up x_ffn_norm in
        let sw = Ops.swiglu gate up in
        Ops.mul_mat w_down sw
      | _ ->
        (* Fallback: simple projection *)
        let wo_alt = match layer.wo with Some t -> t | None -> failwith "wo" in
        Ops.mul_mat wo_alt x_ffn_norm
    in

    (* Residual connection *)
    x := Ops.add !x ffn_out;
  done;

  (* 3. Final Output Norm *)
  let output_norm = match m.weights.output_norm with
    | Some t -> t
    | None -> failwith "Missing output_norm"
  in
  let x_final = Ops.rms_norm !x output_norm eps in

  (* 4. Extract Last Token Vector *)
  let last_col_off = (n_tokens - 1) * n_embd in
  let x_last = Tensor.view_2d x_final last_col_off n_embd 1 n_embd in

  (* 5. Output LM Head Projection *)
  let head = match m.weights.output_head with
    | Some t -> t
    | None -> token_embd
  in
  Ops.mul_mat head x_last

let generate (m : model_instance) ~(prompt : string) ~(max_tokens : int) (cfg : sampler_config)
    ?(on_token : (string -> unit) option) () : string * inference_metrics =
  Kv_cache.clear m.kv_cache;

  (* 1. Encode prompt *)
  let prompt_tokens = Tokenizer.encode m.tokenizer ~bos:true prompt in
  let n_prompt = Array.length prompt_tokens in

  (* 2. Evaluate prompt batch *)
  let t_eval_start = Unix.gettimeofday () in
  let initial_logits = forward m ~token_ids:prompt_tokens ~n_past:0 in
  let t_eval_end = Unix.gettimeofday () in
  let prompt_eval_ms = (t_eval_end -. t_eval_start) *. 1000.0 in

  (* 3. Autoregressive token generation loop *)
  let generated_tokens = ref [] in
  let history_tokens = ref (Array.to_list prompt_tokens) in
  let current_logits = ref initial_logits in
  let n_past = ref n_prompt in

  let t_gen_start = Unix.gettimeofday () in
  let stop = ref false in
  let count = ref 0 in

  while not !stop && !count < max_tokens do
    let next_tid = Sampler.sample cfg ~logits:!current_logits ~prev_tokens:!history_tokens in
    if Some next_tid = m.tokenizer.eos_id then stop := true
    else begin
      generated_tokens := next_tid :: !generated_tokens;
      history_tokens := next_tid :: !history_tokens;
      incr count;

      let piece = Tokenizer.token_to_piece m.tokenizer next_tid in
      (match on_token with Some cb -> cb piece | None -> ());

      (* Forward step for single next token *)
      current_logits := forward m ~token_ids:[| next_tid |] ~n_past:!n_past;
      incr n_past;
    end
  done;

  let t_gen_end = Unix.gettimeofday () in
  let gen_ms = (t_gen_end -. t_gen_start) *. 1000.0 in
  let total_gen = List.length !generated_tokens in
  let tok_sec = if gen_ms > 0.0 then float_of_int total_gen /. (gen_ms /. 1000.0) else 0.0 in

  let generated_text = Tokenizer.decode m.tokenizer (Array.of_list (List.rev !generated_tokens)) in
  let metrics = {
    prompt_tokens = n_prompt;
    gen_tokens = total_gen;
    prompt_eval_ms;
    gen_ms;
    tokens_per_sec = tok_sec;
    peak_memory_mb = float_of_int (m.hp.n_embd * m.hp.n_layer * 4 * 4) /. (1024.0 *. 1024.0);
  } in
  (generated_text, metrics)

let embed (m : model_instance) ~(prompt : string) : float array =
  let tokens = Tokenizer.encode m.tokenizer ~bos:true prompt in
  let n_tokens = Array.length tokens in
  let n_embd = m.hp.n_embd in
  let x = Ops.get_rows (match m.weights.token_embd with Some t -> t | None -> failwith "embd") tokens in
  let out_arr = Array.make n_embd 0.0 in
  match x.data_f32 with
  | Some buf ->
    (* Mean pooling across token representations *)
    for i = 0 to n_embd - 1 do
      let sum = ref 0.0 in
      for t = 0 to n_tokens - 1 do
        sum := !sum +. Array1.unsafe_get buf (t * n_embd + i);
      done;
      out_arr.(i) <- !sum /. float_of_int n_tokens;
    done;
    out_arr
  | None -> out_arr
