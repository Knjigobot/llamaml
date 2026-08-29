(* test_cordis.ml - Cordis-OxCaml Plugin Lifecycle & Service Tests *)

open Cordis_core
open Llamaml

let test_cordis_registration () =
  let ctx = Context.root in
  let unregister = Registry.register ctx (module Cordis_plugin.Plugin) in

  let svc_opt = Service.get ctx "llamaml" in
  Alcotest.(check bool) "llamaml service is available" true (Option.is_some svc_opt);

  (* Unregister plugin -> check revertible effect execution *)
  unregister ();
  let svc_after = Service.get ctx "llamaml" in
  Alcotest.(check bool) "llamaml service cleaned up" true (Option.is_none svc_after)

let () =
  let open Alcotest in
  run "Llamaml.CordisPlugin" [
    "lifecycle", [
      test_case "register_and_cleanup" `Quick test_cordis_registration;
    ];
  ]
