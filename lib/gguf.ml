(* gguf.ml - Complete Binary GGUF v2/v3 Deserializer & Metadata Inspector *)

open Types
open Bigarray

type metadata_val =
  | G_UINT8 of int
  | G_INT8 of int
  | G_UINT16 of int
  | G_INT16 of int
  | G_UINT32 of int32
  | G_INT32 of int32
  | G_FLOAT32 of float
  | G_BOOL of bool
  | G_STRING of string
  | G_ARRAY of metadata_val list
  | G_UINT64 of int64
  | G_INT64 of int64
  | G_FLOAT64 of float

type tensor_info = {
  ti_name : string;
  ti_n_dims : int;
  ti_ne : int array;
  ti_qtype : quant_type;
  ti_offset : int64;
}

type gguf_file = {
  version : int;
  metadata : (string, metadata_val) Hashtbl.t;
  tensor_infos : (string, tensor_info) Hashtbl.t;
  tensor_list : tensor_info list;
  data_offset : int64;
  file_path : string;
  raw_buffer : u8_buffer option;
}

let quant_type_of_int = function
  | 0 -> TYPE_F32
  | 1 -> TYPE_F16
  | 2 -> TYPE_Q4_0
  | 3 -> TYPE_Q4_1
  | 6 -> TYPE_Q5_0
  | 7 -> TYPE_Q5_1
  | 8 -> TYPE_Q8_0
  | 9 -> TYPE_Q8_1
  | 10 -> TYPE_Q2_K
  | 11 -> TYPE_Q3_K
  | 12 -> TYPE_Q4_K
  | 13 -> TYPE_Q5_K
  | 14 -> TYPE_Q6_K
  | 15 -> TYPE_Q8_K
  | 16 -> TYPE_IQ2_XXS
  | 17 -> TYPE_IQ2_XS
  | 18 -> TYPE_IQ3_XXS
  | 19 -> TYPE_IQ1_S
  | 20 -> TYPE_IQ4_NL
  | 21 -> TYPE_IQ4_XS
  | 30 -> TYPE_BF16
  | _ -> TYPE_F32

let read_u8 ic = input_byte ic

let read_u16 ic =
  let b0 = input_byte ic in
  let b1 = input_byte ic in
  b0 lor (b1 lsl 8)

let read_i16 ic =
  let v = read_u16 ic in
  if v >= 32768 then v - 65536 else v

let read_u32 ic =
  let b0 = Int32.of_int (input_byte ic) in
  let b1 = Int32.of_int (input_byte ic) in
  let b2 = Int32.of_int (input_byte ic) in
  let b3 = Int32.of_int (input_byte ic) in
  Int32.logor b0 (Int32.logor (Int32.shift_left b1 8) (Int32.logor (Int32.shift_left b2 16) (Int32.shift_left b3 24)))

let read_i32 ic = read_u32 ic

let read_f32 ic =
  let bits = read_u32 ic in
  Int32.float_of_bits bits

let read_u64 ic =
  let l0 = Int64.of_int32 (read_u32 ic) in
  let l0_pos = Int64.logand l0 0xFFFFFFFFL in
  let l1 = Int64.of_int32 (read_u32 ic) in
  let l1_pos = Int64.logand l1 0xFFFFFFFFL in
  Int64.logor l0_pos (Int64.shift_left l1_pos 32)

let read_i64 ic = read_u64 ic

let read_f64 ic =
  let bits = read_u64 ic in
  Int64.float_of_bits bits

let read_string ic =
  let len = Int64.to_int (read_u64 ic) in
  really_input_string ic len

let rec read_metadata_val ic m_type =
  match m_type with
  | 0 -> G_UINT8 (read_u8 ic)
  | 1 -> G_INT8 (let b = read_u8 ic in if b >= 128 then b - 256 else b)
  | 2 -> G_UINT16 (read_u16 ic)
  | 3 -> G_INT16 (read_i16 ic)
  | 4 -> G_UINT32 (read_u32 ic)
  | 5 -> G_INT32 (read_i32 ic)
  | 6 -> G_FLOAT32 (read_f32 ic)
  | 7 -> G_BOOL (read_u8 ic <> 0)
  | 8 -> G_STRING (read_string ic)
  | 9 ->
    let item_type = read_u32 ic in
    let item_count = Int64.to_int (read_u64 ic) in
    let items = ref [] in
    for _i = 0 to item_count - 1 do
      items := read_metadata_val ic (Int32.to_int item_type) :: !items;
    done;
    G_ARRAY (List.rev !items)
  | 10 -> G_UINT64 (read_u64 ic)
  | 11 -> G_INT64 (read_i64 ic)
  | 12 -> G_FLOAT64 (read_f64 ic)
  | _ -> failwith (Printf.sprintf "Unknown GGUF metadata type %d" m_type)

let parse_file (path : string) : gguf_file =
  let ic = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in ic) (fun () ->
    let magic = really_input_string ic 4 in
    if magic <> "GGUF" then failwith (Printf.sprintf "Invalid GGUF magic: %s" magic);
    let version = Int32.to_int (read_u32 ic) in
    if version < 2 || version > 3 then
      failwith (Printf.sprintf "Unsupported GGUF version %d" version);

    let tensor_count = read_u64 ic in
    let metadata_kv_count = read_u64 ic in

    let metadata = Hashtbl.create 128 in
    for _i = 0 to Int64.to_int metadata_kv_count - 1 do
      let key = read_string ic in
      let val_type = Int32.to_int (read_u32 ic) in
      let v = read_metadata_val ic val_type in
      Hashtbl.replace metadata key v;
    done;

    let tensor_infos = Hashtbl.create 256 in
    let tensor_list = ref [] in
    for _i = 0 to Int64.to_int tensor_count - 1 do
      let name = read_string ic in
      let n_dims = Int32.to_int (read_u32 ic) in
      let ne = [| 1; 1; 1; 1 |] in
      for d = 0 to n_dims - 1 do
        ne.(d) <- Int64.to_int (read_u64 ic);
      done;
      let qtype_raw = Int32.to_int (read_u32 ic) in
      let qtype = quant_type_of_int qtype_raw in
      let offset = read_u64 ic in
      let ti = { ti_name = name; ti_n_dims = n_dims; ti_ne = ne; ti_qtype = qtype; ti_offset = offset } in
      Hashtbl.replace tensor_infos name ti;
      tensor_list := ti :: !tensor_list;
    done;

    (* GGUF data alignment is 32 bytes by default *)
    let current_pos = LargeFile.pos_in ic in
    let alignment =
      match Hashtbl.find_opt metadata "general.alignment" with
      | Some (G_UINT32 a) -> Int64.of_int32 a
      | Some (G_UINT64 a) -> a
      | _ -> 32L
    in
    let rem = Int64.rem current_pos alignment in
    let data_offset = if rem = 0L then current_pos else Int64.add (Int64.sub current_pos rem) alignment in

    let fd = Unix.openfile path [Unix.O_RDONLY] 0o644 in
    let raw_buffer =
      try
        let ba = Array1.map_file fd int8_unsigned c_layout false [| -1 |] in
        Unix.close fd;
        Some (Obj.magic ba)
      with _ ->
        Unix.close fd;
        None
    in

    {
      version;
      metadata;
      tensor_infos;
      tensor_list = List.rev !tensor_list;
      data_offset;
      file_path = path;
      raw_buffer;
    }
  )

let get_metadata_string (f : gguf_file) (k : string) : string option =
  match Hashtbl.find_opt f.metadata k with
  | Some (G_STRING s) -> Some s
  | _ -> None

let get_metadata_int (f : gguf_file) (k : string) : int option =
  match Hashtbl.find_opt f.metadata k with
  | Some (G_UINT32 i) -> Some (Int32.to_int i)
  | Some (G_INT32 i) -> Some (Int32.to_int i)
  | Some (G_UINT64 i) -> Some (Int64.to_int i)
  | Some (G_INT64 i) -> Some (Int64.to_int i)
  | Some (G_UINT16 i) -> Some i
  | Some (G_INT16 i) -> Some i
  | Some (G_UINT8 i) -> Some i
  | Some (G_INT8 i) -> Some i
  | _ -> None

let get_metadata_float (f : gguf_file) (k : string) : float option =
  match Hashtbl.find_opt f.metadata k with
  | Some (G_FLOAT32 fl) -> Some fl
  | Some (G_FLOAT64 fl) -> Some fl
  | _ -> None

let get_metadata_array (f : gguf_file) (k : string) : metadata_val list option =
  match Hashtbl.find_opt f.metadata k with
  | Some (G_ARRAY arr) -> Some arr
  | _ -> None

let extract_hyperparameters (f : gguf_file) : hyperparameters =
  let arch_str = match get_metadata_string f "general.architecture" with
    | Some s -> s
    | None -> "llama"
  in
  let arch = string_to_arch arch_str in
  let prefix = arch_str ^ "." in

  let n_vocab = Option.value (get_metadata_int f (prefix ^ "vocab_size"))
      ~default:(match get_metadata_array f "tokenizer.ggml.tokens" with Some a -> List.length a | None -> 32000) in
  let n_embd = Option.value (get_metadata_int f (prefix ^ "embedding_length")) ~default:4096 in
  let n_layer = Option.value (get_metadata_int f (prefix ^ "block_count")) ~default:32 in
  let n_head = Option.value (get_metadata_int f (prefix ^ "attention.head_count")) ~default:32 in
  let n_head_kv = Option.value (get_metadata_int f (prefix ^ "attention.head_count_kv")) ~default:n_head in
  let n_ff = Option.value (get_metadata_int f (prefix ^ "feed_forward_length")) ~default:(4 * n_embd) in
  let n_rot = Option.value (get_metadata_int f (prefix ^ "rope.dimension_count")) ~default:(n_embd / n_head) in
  let rope_freq_base = Option.value (get_metadata_float f (prefix ^ "rope.freq_base")) ~default:10000.0 in
  let rope_freq_scale = Option.value (get_metadata_float f (prefix ^ "rope.freq_scale")) ~default:1.0 in
  let rms_norm_eps = Option.value (get_metadata_float f (prefix ^ "attention.layer_norm_rms_epsilon")) ~default:1e-5 in
  let n_expert = Option.value (get_metadata_int f (prefix ^ "expert_count")) ~default:0 in
  let n_expert_used = Option.value (get_metadata_int f (prefix ^ "expert_used_count")) ~default:0 in

  {
    arch;
    n_vocab;
    n_embd;
    n_mult = 256;
    n_head;
    n_head_kv;
    n_layer;
    n_rot;
    n_expert;
    n_expert_used;
    n_ff;
    rope_freq_base;
    rope_freq_scale;
    rope_type = 0;
    rms_norm_eps;
    f_clamp_kqv = 0.0;
    f_max_alibi_bias = 0.0;
    expert_weights_scale = 1.0;
  }

let extract_tokenizer (f : gguf_file) : tokenizer =
  let model_type_str = Option.value (get_metadata_string f "tokenizer.ggml.model") ~default:"llama" in
  let t_type = if model_type_str = "gpt2" then TOKENIZER_BPE else TOKENIZER_SPM in

  let vocab = Hashtbl.create 32000 in
  let token_to_id = Hashtbl.create 32000 in
  let merges = Hashtbl.create 16000 in

  (match get_metadata_array f "tokenizer.ggml.tokens" with
   | Some tokens ->
     let scores = match get_metadata_array f "tokenizer.ggml.scores" with
       | Some sc -> sc
       | None -> []
     in
     let token_types = match get_metadata_array f "tokenizer.ggml.token_type" with
       | Some tt -> tt
       | None -> []
     in
     List.iteri (fun idx item ->
       let text = match item with G_STRING s -> s | _ -> "" in
       let score = if idx < List.length scores then (match List.nth scores idx with G_FLOAT32 fl -> fl | _ -> 0.0) else 0.0 in
       let tt = if idx < List.length token_types then (match List.nth token_types idx with G_INT32 i -> Int32.to_int i | _ -> 1) else 1 in
       let data = { text; score; token_type = tt } in
       Hashtbl.replace vocab idx data;
       Hashtbl.replace token_to_id text idx;
     ) tokens
   | None -> ());

  (match get_metadata_array f "tokenizer.ggml.merges" with
   | Some merge_list ->
     List.iteri (fun rank item ->
       match item with
       | G_STRING line ->
         let parts = String.split_on_char ' ' line in
         if List.length parts = 2 then begin
           let p0 = List.nth parts 0 in
           let p1 = List.nth parts 1 in
           Hashtbl.replace merges (p0, p1) rank;
         end
       | _ -> ()
     ) merge_list
   | None -> ());

  let bos_id = get_metadata_int f "tokenizer.ggml.bos_token_id" in
  let eos_id = get_metadata_int f "tokenizer.ggml.eos_token_id" in
  let pad_id = get_metadata_int f "tokenizer.ggml.pad_token_id" in
  let nl_id = get_metadata_int f "tokenizer.ggml.nl_token_id" in
  let unk_id = get_metadata_int f "tokenizer.ggml.unknown_token_id" in

  {
    t_type;
    vocab;
    token_to_id;
    merges;
    bos_id;
    eos_id;
    pad_id;
    nl_id;
    unk_id;
    add_bos = true;
    add_eos = false;
  }

let load_tensor (f : gguf_file) (name : string) : tensor =
  match Hashtbl.find_opt f.tensor_infos name with
  | Some ti ->
    let t = Tensor.create_tensor ~name ti.ti_qtype ti.ti_ne ti.ti_n_dims in
    (match f.raw_buffer with
     | Some raw_file_buf ->
       let abs_byte_off = Int64.to_int (Int64.add f.data_offset ti.ti_offset) in
       let byte_len = Tensor.nbytes t in
       let sub = Array1.sub raw_file_buf abs_byte_off byte_len in
       if ti.ti_qtype = TYPE_F32 then
         t.data_f32 <- Some (Obj.magic sub)
       else
         t.data_raw <- Some sub
     | None ->
       (* Fallback: direct seek and read *)
       let ic = open_in_bin f.file_path in
       LargeFile.seek_in ic (Int64.add f.data_offset ti.ti_offset);
       let byte_len = Tensor.nbytes t in
       let bytes_data = really_input_string ic byte_len in
       close_in ic;
       let buf = Array1.create int8_unsigned c_layout byte_len in
       for i = 0 to byte_len - 1 do
         Array1.unsafe_set buf i (Char.code bytes_data.[i]);
       done;
       if ti.ti_qtype = TYPE_F32 then
         t.data_f32 <- Some (Obj.magic buf)
       else
         t.data_raw <- Some buf);
    t
  | None -> failwith (Printf.sprintf "Tensor %s not found in GGUF file" name)

let load_model_weights (f : gguf_file) (hp : hyperparameters) : model_weights =
  let try_tensor name =
    if Hashtbl.mem f.tensor_infos name then Some (load_tensor f name) else None
  in
  let token_embd = try_tensor "token_embd.weight" in
  let output_norm = try_tensor "output_norm.weight" in
  let output_head = try_tensor "output.weight" in

  let layers = Array.init hp.n_layer (fun l ->
    let pfx = Printf.sprintf "blk.%d." l in
    {
      attn_norm = try_tensor (pfx ^ "attn_norm.weight");
      attn_norm_2 = try_tensor (pfx ^ "attn_norm_2.weight");
      wq = try_tensor (pfx ^ "attn_q.weight");
      wk = try_tensor (pfx ^ "attn_k.weight");
      wv = try_tensor (pfx ^ "attn_v.weight");
      wo = try_tensor (pfx ^ "attn_output.weight");
      ffn_norm = try_tensor (pfx ^ "ffn_norm.weight");
      ffn_norm_2 = try_tensor (pfx ^ "ffn_norm_2.weight");
      w_gate = try_tensor (pfx ^ "ffn_gate.weight");
      w_up = try_tensor (pfx ^ "ffn_up.weight");
      w_down = try_tensor (pfx ^ "ffn_down.weight");
      w_gate_exp = [];
      w_up_exp = [];
      w_down_exp = [];
      mla_q_lora = try_tensor (pfx ^ "attn_q_a.weight");
      mla_kv_lora = try_tensor (pfx ^ "attn_kv_a.weight");
    }
  ) in

  {
    token_embd;
    output_norm;
    output_head;
    layers;
  }
