(* main.ml - Pure OxCaml GGML & LLaMA.cpp CLI Executable *)

open Llamaml
open Llamaml.Types

let usage () =
  Printf.printf "========================================================================\n";
  Printf.printf "  LLAMAML: Industrial-Grade Pure OxCaml Tensor & LLM Engine (OCaml 5+)  \n";
  Printf.printf "  Powered by Cordis-OxCaml Spatiotemporal Meta-Framework               \n";
  Printf.printf "========================================================================\n\n";
  Printf.printf "Usage:\n";
  Printf.printf "  llamaml run     --model <path.gguf> [--prompt <text>] [--max-tokens 128] [--temp 0.7]\n";
  Printf.printf "  llamaml serve   --model <path.gguf> [--port 8092] [--host 127.0.0.1]\n";
  Printf.printf "  llamaml inspect --model <path.gguf>\n";
  Printf.printf "  llamaml bench   --model <path.gguf> [--tokens 100]\n";
  Printf.printf "  llamaml tokenize --model <path.gguf> --text <text>\n\n"

let main () =
  let args = Array.to_list Sys.argv in
  match args with
  | _ :: "run" :: rest ->
    let model_path = ref (if Sys.file_exists "models/Puro-2B-Base.Q4_K_M.gguf" then "models/Puro-2B-Base.Q4_K_M.gguf" else "") in
    let prompt = ref "Explain spatiotemporal composability in pure OxCaml." in
    let max_tokens = ref 128 in
    let temp = ref 0.7 in
    let top_p = ref 0.9 in

    let rec parse = function
      | "--model" :: p :: r -> model_path := p; parse r
      | "--prompt" :: p :: r -> prompt := p; parse r
      | "--max-tokens" :: n :: r -> max_tokens := int_of_string n; parse r
      | "--temp" :: t :: r -> temp := float_of_string t; parse r
      | "--top-p" :: tp :: r -> top_p := float_of_string tp; parse r
      | _ :: r -> parse r
      | [] -> ()
    in
    parse rest;

    if !model_path = "" || not (Sys.file_exists !model_path) then begin
      Printf.eprintf "[Error] Model file not found: %s\n" !model_path;
      exit 1
    end;

    Printf.printf "\n[Llamaml] Loading GGUF Model from %s...\n%!" !model_path;
    let m = Model.load !model_path in
    Printf.printf "[Llamaml] Model Loaded! Arch: %s | Vocab: %d | Embd: %d | Layers: %d | Heads: %d\n"
      (arch_to_string m.hp.arch) m.hp.n_vocab m.hp.n_embd m.hp.n_layer m.hp.n_head;
    Printf.printf "\n>>> Prompt: %s\n\n<<< Generating (Streaming):\n\x1b[32m%!" !prompt;

    let cfg = {
      default_sampler_config with
      temperature = !temp;
      top_p = !top_p;
    } in

    let _, metrics = Model.generate m ~prompt:!prompt ~max_tokens:!max_tokens cfg
        ~on_token:(fun piece -> Printf.printf "%s%!" piece) () in

    Printf.printf "\x1b[0m\n\n------------------------------------------------------------\n";
    Printf.printf "Performance Telemetry:\n";
    Printf.printf "  Prompt Eval Duration : %.2f ms (%d tokens, %.1f tok/s)\n"
      metrics.prompt_eval_ms metrics.prompt_tokens
      (if metrics.prompt_eval_ms > 0.0 then float_of_int metrics.prompt_tokens /. (metrics.prompt_eval_ms /. 1000.0) else 0.0);
    Printf.printf "  Token Gen Duration   : %.2f ms (%d tokens, \x1b[1;33m%.1f tok/s\x1b[0m)\n"
      metrics.gen_ms metrics.gen_tokens metrics.tokens_per_sec;
    Printf.printf "  Active Memory Pool   : %.2f MB\n" metrics.peak_memory_mb;
    Printf.printf "------------------------------------------------------------\n"

  | _ :: "inspect" :: rest ->
    let model_path = ref (if Sys.file_exists "models/Puro-2B-Base.Q4_K_M.gguf" then "models/Puro-2B-Base.Q4_K_M.gguf" else "") in
    let rec parse = function
      | "--model" :: p :: r -> model_path := p; parse r
      | _ :: r -> parse r
      | [] -> ()
    in
    parse rest;

    if !model_path = "" || not (Sys.file_exists !model_path) then begin
      Printf.eprintf "[Error] Model file not found: %s\n" !model_path;
      exit 1
    end;

    Printf.printf "\n[Llamaml-Inspector] Parsing GGUF metadata from %s...\n" !model_path;
    let gguf = Gguf.parse_file !model_path in
    let hp = Gguf.extract_hyperparameters gguf in
    Printf.printf "\nModel Hyperparameters:\n";
    Printf.printf "  Architecture   : %s\n" (arch_to_string hp.arch);
    Printf.printf "  Vocabulary Size: %d\n" hp.n_vocab;
    Printf.printf "  Embedding Dim  : %d\n" hp.n_embd;
    Printf.printf "  Feed Forward   : %d\n" hp.n_ff;
    Printf.printf "  Transformer L  : %d\n" hp.n_layer;
    Printf.printf "  Attention Heads: %d (KV: %d)\n" hp.n_head hp.n_head_kv;
    Printf.printf "  RoPE Freq Base : %.1f\n" hp.rope_freq_base;
    Printf.printf "  Norm Epsilon   : %.2e\n" hp.rms_norm_eps;
    Printf.printf "\nTensor Inventory (%d tensors):\n" (List.length gguf.tensor_list);
    List.iteri (fun idx (ti : Gguf.tensor_info) ->
      if idx < 15 || idx >= List.length gguf.tensor_list - 5 then
        Printf.printf "  [%3d] %-35s %-8s [%d, %d, %d, %d]\n"
          idx ti.ti_name (type_name ti.ti_qtype) ti.ti_ne.(0) ti.ti_ne.(1) ti.ti_ne.(2) ti.ti_ne.(3)
      else if idx = 15 then
        Printf.printf "  ... [%d more tensors] ...\n" (List.length gguf.tensor_list - 20)
    ) gguf.tensor_list

  | _ :: "serve" :: rest ->
    let model_path = ref (if Sys.file_exists "models/Puro-2B-Base.Q4_K_M.gguf" then "models/Puro-2B-Base.Q4_K_M.gguf" else "") in
    let port = ref 8092 in
    let host = ref "127.0.0.1" in
    let rec parse = function
      | "--model" :: p :: r -> model_path := p; parse r
      | "--port" :: p :: r -> port := int_of_string p; parse r
      | "--host" :: h :: r -> host := h; parse r
      | _ :: r -> parse r
      | [] -> ()
    in
    parse rest;

    let s = Server.start {
      port = !port;
      host = !host;
      model_path = !model_path;
      static_dir = None;
    } in
    Server.run_forever s

  | _ :: "bench" :: rest ->
    let model_path = ref (if Sys.file_exists "models/Puro-2B-Base.Q4_K_M.gguf" then "models/Puro-2B-Base.Q4_K_M.gguf" else "") in
    let tokens = ref 50 in
    let rec parse = function
      | "--model" :: p :: r -> model_path := p; parse r
      | "--tokens" :: n :: r -> tokens := int_of_string n; parse r
      | _ :: r -> parse r
      | [] -> ()
    in
    parse rest;

    Printf.printf "\n[Llamaml Benchmark] Running %d tokens speed benchmark on %s...\n" !tokens !model_path;
    let m = Model.load !model_path in
    let prompt = "The category-theoretic foundation of deep learning computation graphs is" in
    let _, metrics = Model.generate m ~prompt ~max_tokens:!tokens default_sampler_config () in
    Printf.printf "\nBenchmark Result: \x1b[1;32m%.2f tokens/second\x1b[0m\n" metrics.tokens_per_sec

  | _ -> usage ()

let () = main ()
