# Llamaml: Pure OxCaml GGML Tensor & LLaMA.cpp Engine

[![CI](https://github.com/Knjigobot/Cordis-OxCaml/actions/workflows/ci.yml/badge.svg)](https://github.com/Knjigobot/Cordis-OxCaml/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![OCaml 5+](https://img.shields.io/badge/OCaml-5.0%2B-orange.svg)](https://ocaml.org)
[![Zero Python](https://img.shields.io/badge/Python-0%25-green.svg)](#)
[![Zero C/C++](https://img.shields.io/badge/C%2FC%2B%2B-0%25-green.svg)](#)
[![Cordis Plugin](https://img.shields.io/badge/Cordis-Native%20Plugin-purple.svg)](#)

**Llamaml** is an industrial-grade, mathematically verified, full-fidelity port of **ggml** (tensor computation engine) and **llama.cpp** (LLM inference runtime) in native **OxCaml** (OCaml 5+ with Algebraic Effects and Jane Street unboxed primitives).

Built as a first-class native subsystem powered by the **[Cordis-OxCaml](https://github.com/Knjigobot/Cordis-OxCaml)** Spatiotemporal Composability Meta-Framework, Llamaml achieves zero-leak memory safety, delimited speculative rollbacks, unboxed Bigarray SIMD tensor compute, and zero process restart.

---

## 🏛 System Topography

```
+---------------------------------------------------------------------------------------------------+
|                                      LLAMAML RUNTIME (OxCaml 5+)                                  |
+---------------------------------------------------------------------------------------------------+
|  CORDIS-OXCAML HOST KERNEL                                                                        |
|  - Spatiotemporal Coeffect Manifold (Context, Scope, Dynamic Services)                           |
|  - Revertible LIFO Effect Disposal (KV Cache & Weight Lifetime Management)                        |
|  - OCaml 5 Algebraic Effects (Effect.Deep Delimited Continuations for Speculative Rollback)       |
+---------------------------------------------------------------------------------------------------+
|  GGML TENSOR COMPUTATION ENGINE (llamaml/lib/quant, ops, cgraph, alloc)                          |
|  - Full Quantization Kernels (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, F16, F32)  |
|  - Unboxed Bigarray Tensors & Block-Vectorized Dot Products                                       |
|  - Computation Graph DAG with Topological Scheduling & Operator Fusion                            |
|  - Virtual Buffer Lifetime Analysis & Scratch Ring Buffers (Zero GC Allocation)                   |
|  - FlashAttention-2 (Online Tiled Softmax) & RoPE (NeOX/LLaMA/YaRN Scaling)                       |
+---------------------------------------------------------------------------------------------------+
|  LLAMA.CPP INFERENCE & TRANSFORMER ENGINE (llamaml/lib/gguf, tokenizer, model, sampler)          |
|  - Zero-Copy GGUF v2/v3 Binary Deserializer & Metadata Inspector                                  |
|  - BPE, SentencePiece, Unigram Tokenizers with Byte Fallback & UTF-8 Stream Decoding               |
|  - Universal Transformer (LLaMA 1-3.2, Mistral, Mixtral MoE, Qwen2.5, DeepSeek MLA/MoE, Gemma)   |
|  - Paged / Ring Buffer KV Cache Management with Prefix Caching & Dynamic Context Shifting         |
|  - Full Sampling Suite (Greedy, Temp, Top-K, Top-P, Min-P, TFS, Mirostat, DRY, GBNF Grammar)     |
+---------------------------------------------------------------------------------------------------+
|  INTERFACES & FORMAL VERIFICATION                                                                 |
|  - OpenAI-Compatible HTTP / Server-Sent Events (SSE) Server (/v1/chat/completions, /v1/models)    |
|  - Rich CLI Terminal REPL & Live Performance Telemetry (tok/s, prompt eval ms, memory footprint)  |
|  - High-Performance Web Dashboard & Computation Graph Visualizer                                  |
|  - Formal Verification: Cubical Agda (Tensor Invariants) & Rzk (DAG Homotopy & LIFO Rollbacks)   |
+---------------------------------------------------------------------------------------------------+
```

---

## 🔬 Core Architectural Guarantees

1. **Zero-C / Zero-Python Pure OxCaml**: The entire tensor runtime, GGUF binary parser, quantization decoders, matrix multiply kernels, and tokenizers are written 100% in OCaml 5+.
2. **Cordis Spatiotemporal Coeffects & Revertible LIFO Disposal**: Model weights, KV cache buffers, and async SSE streaming channels are scoped to Cordis `Scope.t`. When a plugin or session terminates, all resources unwind in strict reverse topological order with zero memory leaks.
3. **Speculative Decoding via Delimited Continuations**: OCaml 5 `Effect.Deep` delimited continuations allow the inference engine to speculatively branch along multiple draft trajectories and instantaneously revert discarded token branches.
4. **Unboxed SIMD Bigarrays**: High-speed GEMM kernels operate directly on aligned `Bigarray.Array1.t` blocks without GC nursery allocation overhead.
5. **Formal Verification (Agda & Rzk)**: Verified in Cubical Agda (`formal/agda/LlamamlTensor.agda`) and Synthetic Simplicial Homotopy Type Theory (`formal/rzk/LlamamlHomotopy.rzk`).

---

## 🚀 Quickstart & Usage

### 1. Build with Dune
```bash
# Build all libraries, CLI executables, and tests
dune build @all

# Run the complete automated verification test suite
dune runtest
```

### 2. CLI Inference & Streaming
```bash
# Run streaming text generation on local GGUF model
dune exec bin/main.exe -- run --model models/Puro-2B-Base.Q4_K_M.gguf --prompt "Explain spatiotemporal composability." --temp 0.7

# Inspect GGUF header, architecture, hyperparameters, and tensor inventory
dune exec bin/main.exe -- inspect --model models/Puro-2B-Base.Q4_K_M.gguf

# Run speed benchmark
dune exec bin/main.exe -- bench --model models/Puro-2B-Base.Q4_K_M.gguf --tokens 100
```

### 3. OpenAI-Compatible Server & Interactive Web UI
```bash
# Start the native HTTP/SSE inference daemon on port 8092
dune exec bin/main.exe -- serve --model models/Puro-2B-Base.Q4_K_M.gguf --port 8092

# Or launch via Windows batch script:
./RUN.bat
```
Navigate to `http://localhost:8092` for the interactive visual web dashboard, GGUF metadata inspector, computation DAG visualizer, and live token probability playground.

---

## 📜 Attributions & Licensing

Released under the **MIT License**.

Attributions:
* **GGML & LLaMA.cpp**: Developed by [Georgi Gerganov](https://github.com/ggerganov) and the [ggml.ai](https://ggml.ai) team.
* **Cordis Meta-Framework**: Developed by [Shigma](https://github.com/shigma) & [Cordiverse](https://github.com/cordiverse/cordis).
* **OxCaml & Jane Street Tools**: Jane Street Group LLC.
