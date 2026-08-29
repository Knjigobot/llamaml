(* ops.mli - Mathematical Tensor Operators & High-Performance Kernels *)

open Types

val add : tensor -> tensor -> tensor
val sub : tensor -> tensor -> tensor
val mul : tensor -> tensor -> tensor
val scale : tensor -> float -> tensor

val silu : tensor -> tensor
val gelu : tensor -> tensor
val gelu_quick : tensor -> tensor
val relu : tensor -> tensor
val sigmoid : tensor -> tensor
val swiglu : tensor -> tensor -> tensor

val rms_norm : tensor -> tensor -> float -> tensor
val layer_norm : tensor -> tensor -> tensor option -> float -> tensor

val soft_max : tensor -> tensor
val soft_max_ext : tensor -> float -> bool -> tensor

val rope : tensor -> int -> int -> int -> float -> float -> int -> tensor

val mul_mat : tensor -> tensor -> tensor
val flash_attention_2 : tensor -> tensor -> tensor -> float -> bool -> tensor
val get_rows : tensor -> int array -> tensor
val moe_gate : tensor -> int -> (int array * float array) array
