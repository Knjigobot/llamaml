(lang dune 3.10)
(name llamaml)
(version 1.0.0)

(generate_opam_files true)

(license MIT)
(authors "DeepSeek-AI" "Knjigobot" "Jane Street" "Llamaml Contributors")
(maintainers "llamaml@cordis.org")

(package
 (name llamaml)
 (synopsis "Pure OxCaml GGML Tensor Engine & LLaMA.cpp LLM Inference on Cordis-OxCaml")
 (description
  "Llamaml is an industrial-grade, full-fidelity port of ggml and llama.cpp in pure native OxCaml (OCaml 5+ with Algebraic Effects). Powered by the Cordis-OxCaml Spatiotemporal Composability Meta-Framework, it delivers zero-GC unboxed Bigarray tensor compute, full quantization kernels (Q4_0, Q4_K, Q8_0, F16), GGUF v2/v3 binary parsing, universal Transformer architecture (LLaMA, Mistral, Mixtral, Qwen, DeepSeek MLA/MoE), FlashAttention-2, advanced sampling, OpenAI-compatible HTTP/SSE server, and formal verification in Cubical Agda and Rzk.")
 (depends
  (ocaml (>= 5.0.0))
  (dune (>= 3.10.0))
  (unix (>= 0))
  (bigarray (>= 0))
  (alcotest :with-test))
 (tags ("llamaml" "ggml" "llama.cpp" "llm" "oxcaml" "cordis" "effects" "quantization" "inference" "formal-verification")))
