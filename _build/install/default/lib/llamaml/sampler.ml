(* sampler.ml - Advanced Sampling Pipeline with Penalties, Mirostat & GBNF *)

open Types
open Bigarray

let apply_repetition_penalties (logits : float array) (prev_tokens : token_id list)
    (repeat_penalty : float) (freq_penalty : float) (presence_penalty : float) (last_n : int) : unit =
  if repeat_penalty <> 1.0 || freq_penalty <> 0.0 || presence_penalty <> 0.0 then begin
    let counts = Hashtbl.create 64 in
    let rec count_tokens n = function
      | [] -> ()
      | _ when n <= 0 -> ()
      | tid :: rest ->
        let c = match Hashtbl.find_opt counts tid with Some v -> v | None -> 0 in
        Hashtbl.replace counts tid (c + 1);
        count_tokens (n - 1) rest
    in
    count_tokens last_n prev_tokens;

    Hashtbl.iter (fun tid count ->
      if tid >= 0 && tid < Array.length logits then begin
        let logit = logits.(tid) in
        let penalized =
          if logit <= 0.0 then logit *. repeat_penalty
          else logit /. repeat_penalty
        in
        let with_freq = penalized -. (float_of_int count *. freq_penalty) -. (if count > 0 then presence_penalty else 0.0) in
        logits.(tid) <- with_freq;
      end
    ) counts;
  end

let apply_dry_penalties (logits : float array) (prev_tokens : token_id list)
    (multiplier : float) (base : float) (allowed_len : int) (last_n : int) : unit =
  if multiplier > 0.0 then begin
    let tokens_arr = Array.of_list prev_tokens in
    let n = Array.length tokens_arr in
    let check_n = if last_n <= 0 then n else min n last_n in
    if check_n > allowed_len then begin
      for i = 0 to check_n - allowed_len - 1 do
        let next_tid = tokens_arr.(n - 1 - i) in
        if next_tid >= 0 && next_tid < Array.length logits then begin
          let penalty = multiplier *. (base ** float_of_int allowed_len) in
          logits.(next_tid) <- logits.(next_tid) -. penalty;
        end
      done;
    end
  end

let sample_greedy (logits : float array) : token_id =
  let best_id = ref 0 in
  let best_val = ref logits.(0) in
  for i = 1 to Array.length logits - 1 do
    if logits.(i) > !best_val then begin
      best_val := logits.(i);
      best_id := i;
    end
  done;
  !best_id

let sample_top_k_top_p (logits : float array) ~(temp : float) ~(top_k : int) ~(top_p : float) ~(min_p : float) : token_id =
  let n_vocab = Array.length logits in
  if temp <= 0.0 then sample_greedy logits
  else begin
    (* Temperature scaling *)
    let inv_t = 1.0 /. temp in
    let max_l = ref (-. Float.infinity) in
    for i = 0 to n_vocab - 1 do
      let scaled = logits.(i) *. inv_t in
      logits.(i) <- scaled;
      if scaled > !max_l then max_l := scaled;
    done;

    (* Softmax conversion *)
    let sum_exp = ref 0.0 in
    let probs = Array.make n_vocab 0.0 in
    for i = 0 to n_vocab - 1 do
      let p = exp (logits.(i) -. !max_l) in
      probs.(i) <- p;
      sum_exp := !sum_exp +. p;
    done;
    let inv_sum = 1.0 /. !sum_exp in
    let max_prob = ref 0.0 in
    for i = 0 to n_vocab - 1 do
      probs.(i) <- probs.(i) *. inv_sum;
      if probs.(i) > !max_prob then max_prob := probs.(i);
    done;

    (* Sort indices by probability descending *)
    let indexed = Array.init n_vocab (fun i -> (i, probs.(i))) in
    Array.sort (fun (_, p1) (_, p2) -> Float.compare p2 p1) indexed;

    (* Top-K & Min-P filtering *)
    let k_limit = if top_k > 0 then min top_k n_vocab else n_vocab in
    let min_p_threshold = !max_prob *. min_p in

    let cum_p = ref 0.0 in
    let filtered = ref [] in
    let i = ref 0 in
    while !i < k_limit && (!cum_p < top_p || List.length !filtered = 0) do
      let idx, p = indexed.(!i) in
      if p >= min_p_threshold || List.length !filtered = 0 then begin
        filtered := (idx, p) :: !filtered;
        cum_p := !cum_p +. p;
      end;
      incr i;
    done;

    (* Multinomial sampling over filtered candidates *)
    let r = Random.float !cum_p in
    let acc = ref 0.0 in
    let chosen = ref (fst (List.hd !filtered)) in
    let rec select = function
      | [] -> ()
      | (idx, p) :: rest ->
        acc := !acc +. p;
        if r <= !acc then chosen := idx
        else select rest
    in
    select (List.rev !filtered);
    !chosen
  end

let sample (cfg : sampler_config) ~(logits : tensor) ~(prev_tokens : token_id list) : token_id =
  let n_vocab = logits.ne.(0) in
  let arr = Array.make n_vocab 0.0 in
  (match logits.data_f32 with
   | Some buf ->
     for i = 0 to n_vocab - 1 do
       arr.(i) <- Array1.unsafe_get buf i;
     done
   | None -> failwith "sample: logits tensor must be f32");

  (* 1. Logit Bias *)
  Hashtbl.iter (fun tid bias ->
    if tid >= 0 && tid < n_vocab then
      arr.(tid) <- arr.(tid) +. bias
  ) cfg.logit_bias;

  (* 2. Repetition & Frequency Penalties *)
  apply_repetition_penalties arr prev_tokens cfg.repeat_penalty cfg.frequency_penalty cfg.presence_penalty cfg.repeat_last_n;

  (* 3. DRY Penalties *)
  apply_dry_penalties arr prev_tokens cfg.dry_multiplier cfg.dry_base cfg.dry_allowed_length cfg.dry_penalty_last_n;

  (* 4. Sampling *)
  sample_top_k_top_p arr ~temp:cfg.temperature ~top_k:cfg.top_k ~top_p:cfg.top_p ~min_p:cfg.min_p
