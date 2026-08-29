(* tensor.mli - Multidimensional Bigarray Tensor Manipulation *)

open Types
open Bigarray

val create_1d : ?name:string -> quant_type -> int -> tensor
val create_2d : ?name:string -> quant_type -> int -> int -> tensor
val create_3d : ?name:string -> quant_type -> int -> int -> int -> tensor
val create_4d : ?name:string -> quant_type -> int -> int -> int -> int -> tensor

val from_f32_array : ?name:string -> int array -> float array -> tensor
val to_f32_array : tensor -> float array

val nelements : tensor -> int
val nbytes : tensor -> int

val get_f32_1d : tensor -> int -> float
val set_f32_1d : tensor -> int -> float -> unit

val get_f32_2d : tensor -> int -> int -> float
val set_f32_2d : tensor -> int -> int -> float -> unit

val view_1d : tensor -> int -> int -> tensor
val view_2d : tensor -> int -> int -> int -> int -> tensor
val reshape : tensor -> int array -> tensor

val copy_f32 : tensor -> tensor -> unit
val zero : tensor -> unit
