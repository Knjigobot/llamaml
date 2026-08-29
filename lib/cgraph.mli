(* cgraph.mli - Computation Graph DAG Construction & Evaluation Engine *)

open Types

val create : unit -> cgraph
val add_node : cgraph -> tensor -> unit
val add_leaf : cgraph -> tensor -> unit

val build_forward_expand : cgraph -> tensor -> unit
val topological_sort : cgraph -> tensor list
val eval : cgraph -> unit
val reset : cgraph -> unit

val to_mermaid : cgraph -> string
val to_dot : cgraph -> string
