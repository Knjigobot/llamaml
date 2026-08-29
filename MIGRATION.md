# Migration Guide: C/C++ ggml & llama.cpp to Pure OxCaml Llamaml

This guide helps developers migrating from `llama.cpp` / `ggml` C/C++ libraries to native `Llamaml` in OxCaml.

---

## 1. Concept Mapping

| C/C++ (`ggml` / `llama.cpp`) | Pure OxCaml (`Llamaml`) | Notes |
| :--- | :--- | :--- |
| `struct ggml_tensor` | `Llamaml.Types.tensor` | Unboxed Bigarray representation |
| `struct ggml_cgraph` | `Llamaml.Cgraph.t` | DAG with topological sort |
| `ggml_build_forward_expand` | `Llamaml.Cgraph.build_forward_expand` | Functional graph builder |
| `ggml_graph_compute_with_ctx` | `Llamaml.Cgraph.eval` | Multi-domain parallel execution |
| `ggml_alloc` | `Llamaml.Alloc.plan_graph_memory` | Static lifetime interval coloring |
| `llama_model` | `Llamaml.Model.model_instance` | Immutable model record |
| `llama_decode` | `Llamaml.Model.forward` | Algebraic effect-aware forward step |
| `llama_sample_token` | `Llamaml.Sampler.sample` | Pipeline with Top-P, Min-P, Mirostat |
| `llama_context` / KV Cache | `Llamaml.Kv_cache.t` | Paged ring-buffer cache |
| `ggml_backend_t` / `CUDA` / `Metal` | `Cordis_core.Context.t` | GADT coeffect manifold |

---

## 2. Example: Running Model Inference

### In C++ (`llama.cpp`):
```cpp
llama_model_params model_params = llama_model_default_params();
llama_model * model = llama_load_model_from_file("model.gguf", model_params);
llama_context_params ctx_params = llama_context_default_params();
llama_context * ctx = llama_new_context_with_model(model, ctx_params);

llama_batch batch = llama_batch_init(512, 0, 1);
// ... manual batch pushing and memory freeing ...
llama_free(ctx);
llama_free_model(model);
```

### In Pure OxCaml (`Llamaml` on `Cordis-OxCaml`):
```ocaml
open Llamaml

let () =
  let model = Model.load "models/Puro-2B-Base.Q4_K_M.gguf" in
  let cfg = Types.default_sampler_config in
  let text, metrics = Model.generate model ~prompt:"Hello OxCaml" ~max_tokens:64 cfg () in
  Printf.printf "Generated: %s (%.1f tok/s)\n" text metrics.tokens_per_sec
```
When using the Cordis plugin, all memory, caches, and weights are automatically reclaimed via LIFO revertible effects upon context termination!
