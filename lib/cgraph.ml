(* cgraph.ml - Computation Graph DAG Construction & Evaluation Engine *)

open Types

let create () : cgraph =
  {
    nodes = [];
    leafs = [];
    n_nodes = 0;
    n_leafs = 0;
  }

let add_node (g : cgraph) (t : tensor) : unit =
  g.nodes <- t :: g.nodes;
  g.n_nodes <- g.n_nodes + 1

let add_leaf (g : cgraph) (t : tensor) : unit =
  g.leafs <- t :: g.leafs;
  g.n_leafs <- g.n_leafs + 1

let build_forward_expand (g : cgraph) (root : tensor) : unit =
  let visited = Hashtbl.create 64 in
  let rec traverse (t : tensor) =
    if not (Hashtbl.mem visited t.id) then begin
      Hashtbl.add visited t.id true;
      (match t.src0 with Some s0 -> traverse s0 | None -> ());
      (match t.src1 with Some s1 -> traverse s1 | None -> ());
      (match t.src2 with Some s2 -> traverse s2 | None -> ());
      if t.op = OP_NONE then add_leaf g t
      else add_node g t
    end
  in
  traverse root;
  g.nodes <- List.rev g.nodes;
  g.leafs <- List.rev g.leafs

let topological_sort (g : cgraph) : tensor list =
  List.rev g.nodes

let eval (g : cgraph) : unit =
  let order = topological_sort g in
  List.iter (fun (t : tensor) ->
    match t.op with
    | OP_NONE -> ()
    | OP_ADD ->
      (match (t.src0, t.src1) with
       | Some a, Some b ->
         let res = Ops.add a b in
         t.data_f32 <- res.data_f32
       | _ -> failwith "OP_ADD missing src")
    | OP_SUB ->
      (match (t.src0, t.src1) with
       | Some a, Some b ->
         let res = Ops.sub a b in
         t.data_f32 <- res.data_f32
       | _ -> failwith "OP_SUB missing src")
    | OP_MUL ->
      (match (t.src0, t.src1) with
       | Some a, Some b ->
         let res = Ops.mul a b in
         t.data_f32 <- res.data_f32
       | _ -> failwith "OP_MUL missing src")
    | OP_SILU ->
      (match t.src0 with
       | Some a ->
         let res = Ops.silu a in
         t.data_f32 <- res.data_f32
       | None -> failwith "OP_SILU missing src0")
    | OP_GELU ->
      (match t.src0 with
       | Some a ->
         let res = Ops.gelu a in
         t.data_f32 <- res.data_f32
       | None -> failwith "OP_GELU missing src0")
    | OP_RMS_NORM ->
      (match (t.src0, t.src1) with
       | Some a, Some w ->
         let eps = 1e-5 in
         let res = Ops.rms_norm a w eps in
         t.data_f32 <- res.data_f32
       | _ -> failwith "OP_RMS_NORM missing src")
    | OP_MUL_MAT ->
      (match (t.src0, t.src1) with
       | Some w, Some a ->
         let res = Ops.mul_mat w a in
         t.data_f32 <- res.data_f32
       | _ -> failwith "OP_MUL_MAT missing src")
    | OP_SOFT_MAX ->
      (match t.src0 with
       | Some a ->
         let res = Ops.soft_max a in
         t.data_f32 <- res.data_f32
       | None -> failwith "OP_SOFT_MAX missing src0")
    | OP_SWIGLU ->
      (match (t.src0, t.src1) with
       | Some gate, Some up ->
         let res = Ops.swiglu gate up in
         t.data_f32 <- res.data_f32
       | _ -> failwith "OP_SWIGLU missing src")
    | _ -> ()
  ) order

let reset (g : cgraph) : unit =
  g.nodes <- [];
  g.leafs <- [];
  g.n_nodes <- 0;
  g.n_leafs <- 0

let to_mermaid (g : cgraph) : string =
  let buf = Buffer.create 512 in
  Buffer.add_string buf "graph TD\n";
  List.iter (fun (leaf : tensor) ->
    Buffer.add_string buf (Printf.sprintf "  N%d[\"%s<br/>%s %s\"]\n"
      leaf.id leaf.name (type_name leaf.qtype)
      (Printf.sprintf "[%d,%d]" leaf.ne.(0) leaf.ne.(1)))
  ) g.leafs;
  List.iter (fun (node : tensor) ->
    Buffer.add_string buf (Printf.sprintf "  N%d{\"%s: %s<br/>%s\"}\n"
      node.id node.name (op_name node.op)
      (Printf.sprintf "[%d,%d]" node.ne.(0) node.ne.(1)));
    (match node.src0 with
     | Some s0 -> Buffer.add_string buf (Printf.sprintf "  N%d --> N%d\n" s0.id node.id)
     | None -> ());
    (match node.src1 with
     | Some s1 -> Buffer.add_string buf (Printf.sprintf "  N%d --> N%d\n" s1.id node.id)
     | None -> ());
    (match node.src2 with
     | Some s2 -> Buffer.add_string buf (Printf.sprintf "  N%d --> N%d\n" s2.id node.id)
     | None -> ())
  ) g.nodes;
  Buffer.contents buf

let to_dot (g : cgraph) : string =
  let buf = Buffer.create 512 in
  Buffer.add_string buf "digraph G {\n  rankdir=TB;\n";
  List.iter (fun (node : tensor) ->
    Buffer.add_string buf (Printf.sprintf "  node_%d [label=\"%s (%s)\"];\n" node.id node.name (op_name node.op));
    (match node.src0 with
     | Some s0 -> Buffer.add_string buf (Printf.sprintf "  node_%d -> node_%d;\n" s0.id node.id)
     | None -> ());
    (match node.src1 with
     | Some s1 -> Buffer.add_string buf (Printf.sprintf "  node_%d -> node_%d;\n" s1.id node.id)
     | None -> ())
  ) g.nodes;
  Buffer.add_string buf "}\n";
  Buffer.contents buf
