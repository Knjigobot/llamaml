(* tokenizer.ml - High-Fidelity BPE & SentencePiece Tokenizer *)

open Types

let create ?(t_type = TOKENIZER_SPM) () : tokenizer =
  {
    t_type;
    vocab = Hashtbl.create 32000;
    token_to_id = Hashtbl.create 32000;
    merges = Hashtbl.create 16000;
    bos_id = Some 1;
    eos_id = Some 2;
    pad_id = None;
    nl_id = Some 13;
    unk_id = Some 0;
    add_bos = true;
    add_eos = false;
  }

let add_token (tok : tokenizer) (id : token_id) (text : string) (score : float) (t_type : int) : unit =
  let data = { text; score; token_type = t_type } in
  Hashtbl.replace tok.vocab id data;
  Hashtbl.replace tok.token_to_id text id

let add_merge (tok : tokenizer) (p0 : string) (p1 : string) (rank : int) : unit =
  Hashtbl.replace tok.merges (p0, p1) rank

let token_to_piece (tok : tokenizer) (id : token_id) : string =
  match Hashtbl.find_opt tok.vocab id with
  | Some d -> d.text
  | None -> ""

let piece_to_token (tok : tokenizer) (piece : string) : token_id option =
  Hashtbl.find_opt tok.token_to_id piece

let is_special_token (tok : tokenizer) (id : token_id) : bool =
  match Hashtbl.find_opt tok.vocab id with
  | Some d -> d.token_type = 3 || d.token_type = 2
  | None -> false

(* BPE merge procedure *)
let bpe_encode_word (tok : tokenizer) (word : string) : string list =
  let pieces = ref (List.init (String.length word) (fun i -> String.make 1 word.[i])) in
  let changed = ref true in
  while !changed && List.length !pieces > 1 do
    changed := false;
    let min_rank = ref max_int in
    let best_pair = ref ("", "") in
    let best_idx = ref (-1) in

    let rec find_best idx = function
      | p0 :: (p1 :: _ as rest) ->
        (match Hashtbl.find_opt tok.merges (p0, p1) with
         | Some rank when rank < !min_rank ->
           min_rank := rank;
           best_pair := (p0, p1);
           best_idx := idx
         | _ -> ());
        find_best (idx + 1) rest
      | _ -> ()
    in
    find_best 0 !pieces;

    if !best_idx >= 0 then begin
      changed := true;
      let new_pieces = ref [] in
      let rec apply_merge idx = function
        | p0 :: p1 :: rest when idx = !best_idx ->
          new_pieces := (p0 ^ p1) :: !new_pieces;
          List.iter (fun p -> new_pieces := p :: !new_pieces) rest
        | p :: rest ->
          new_pieces := p :: !new_pieces;
          apply_merge (idx + 1) rest
        | [] -> ()
      in
      apply_merge 0 !pieces;
      pieces := List.rev !new_pieces;
    end
  done;
  !pieces

let encode (tok : tokenizer) ?(bos = tok.add_bos) ?(eos = tok.add_eos) (text : string) : token_id array =
  let tokens = ref [] in
  if bos then (match tok.bos_id with Some id -> tokens := id :: !tokens | None -> ());

  if text <> "" then begin
    (* Pre-processing: replace spaces with SentencePiece marker   if SPM *)
    let formatted =
      if tok.t_type = TOKENIZER_SPM then
        " " ^ (String.concat " " (String.split_on_char ' ' text))
      else text
    in

    (* Greedy / BPE tokenization *)
    let n = String.length formatted in
    let i = ref 0 in
    while !i < n do
      (* Check longest match first *)
      let matched = ref false in
      let max_len = min 64 (n - !i) in
      let l = ref max_len in
      while !l > 0 && not !matched do
        let sub = String.sub formatted !i !l in
        match Hashtbl.find_opt tok.token_to_id sub with
        | Some tid ->
          tokens := tid :: !tokens;
          i := !i + !l;
          matched := true
        | None ->
          decr l
      done;
      if not !matched then begin
        (* Byte fallback: <0xXX> *)
        let byte_code = Char.code formatted.[!i] in
        let hex_piece = Printf.sprintf "<0x%02X>" byte_code in
        (match Hashtbl.find_opt tok.token_to_id hex_piece with
         | Some tid -> tokens := tid :: !tokens
         | None ->
           (match tok.unk_id with
            | Some uid -> tokens := uid :: !tokens
            | None -> ()));
        incr i;
      end
    done;
  end;

  if eos then (match tok.eos_id with Some id -> tokens := id :: !tokens | None -> ());
  Array.of_list (List.rev !tokens)

let decode (tok : tokenizer) (tokens : token_id array) : string =
  let buf = Buffer.create (Array.length tokens * 4) in
  for i = 0 to Array.length tokens - 1 do
    let tid = tokens.(i) in
    let piece = token_to_piece tok tid in
    if not (is_special_token tok tid) then begin
      if String.length piece >= 6 && String.sub piece 0 3 = "<0x" && piece.[5] = '>' then begin
        let hex_str = String.sub piece 3 2 in
        try
          let byte_val = int_of_string ("0x" ^ hex_str) in
          Buffer.add_char buf (Char.chr byte_val)
        with _ ->
          Buffer.add_string buf piece
      end else begin
        (* Replace SPM marker   with space *)
        let clean = ref "" in
        let n = String.length piece in
        let j = ref 0 in
        while !j < n do
          if !j + 2 < n && piece.[!j] = '\xe2' && piece.[!j+1] = '\x96' && piece.[!j+2] = '\x81' then begin
            clean := !clean ^ " ";
            j := !j + 3;
          end else begin
            clean := !clean ^ String.make 1 piece.[!j];
            incr j;
          end
        done;
        Buffer.add_string buf !clean;
      end
    end
  done;
  Buffer.contents buf
