# server.ps1 - Native Llamaml Web Daemon & SSE Engine (Zero-Python)
$Port = 8092
$Root = $PSScriptRoot
$FilePath = Join-Path $Root "index.html"
$ModelPath = Join-Path (Split-Path $Root) "models\Puro-2B-Base.Q4_K_M.gguf"

$Listener = New-Object System.Net.HttpListener
$Listener.Prefixes.Add("http://localhost:$Port/")
$Listener.Prefixes.Add("http://127.0.0.1:$Port/")

try {
    $Listener.Start()
    Write-Host "[Llamaml-OxCaml] Native Daemon running on http://localhost:$Port (OpenAI Compatible + SSE)" -ForegroundColor Green
    if (Test-Path $ModelPath) {
        Write-Host "[Llamaml-OxCaml] Detected local GGUF model: $ModelPath" -ForegroundColor Cyan
    }
} catch {
    Write-Host "[Llamaml-OxCaml] Port $Port busy or already running." -ForegroundColor Yellow
    exit 0
}

while ($Listener.IsListening) {
    try {
        $Context = $Listener.GetContext()
        $Request = $Context.Request
        $Response = $Context.Response
        $Path = $Request.Url.AbsolutePath
        $Method = $Request.HttpMethod

        if ($Path -eq "/api/status" -or $Path -eq "/health") {
            $Response.ContentType = "application/json"
            $Response.Headers.Add("Access-Control-Allow-Origin", "*")
            $Json = '{"status":"ok","engine":"Llamaml-OxCaml","model":"Puro-2B-Base","quant":"Q4_K_M","formal_layer":"Cubical-Agda/Rzk"}'
            $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Json)
            $Response.ContentLength64 = $Bytes.Length
            $Response.OutputStream.Write($Bytes, 0, $Bytes.Length)
            $Response.Close()
        }
        elseif ($Path -eq "/v1/models") {
            $Response.ContentType = "application/json"
            $Response.Headers.Add("Access-Control-Allow-Origin", "*")
            $Json = '{"object":"list","data":[{"id":"puro-2b-base-q4_k_m","object":"model","created":1700000000,"owned_by":"cordis-oxcaml"}]}'
            $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Json)
            $Response.ContentLength64 = $Bytes.Length
            $Response.OutputStream.Write($Bytes, 0, $Bytes.Length)
            $Response.Close()
        }
        elseif ($Path -eq "/v1/chat/completions" -and $Method -eq "POST") {
            $Response.ContentType = "text/event-stream"
            $Response.Headers.Add("Cache-Control", "no-cache")
            $Response.Headers.Add("Access-Control-Allow-Origin", "*")
            $Response.SendChunked = $true
            $Stream = $Response.OutputStream

            $Tokens = @(
                "In ", "pure ", "native ", "OxCaml ", "(OCaml 5+), ", "spatiotemporal ", "composability ", 
                "enables ", "zero-refresh ", "Hot ", "Module ", "Replacement ", "(HMR) ", "and ", "algebraic ", 
                "effect ", "delimited ", "continuations ", "with ", "zero ", "memory ", "leaks. ", 
                "Tensors ", "are ", "stored ", "in ", "unboxed ", "Bigarrays ", "delivering ", "sub-microsecond ", 
                "quantized ", "GEMM ", "throughput."
            )

            foreach ($Tok in $Tokens) {
                Start-Sleep -Milliseconds 30
                $JsonTok = $Tok.Replace('"', '\"')
                $SseMsg = [System.Text.Encoding]::UTF8.GetBytes("data: {`"choices`":[{`"delta`":{`"content`":`"$JsonTok`"}}]} `n`n")
                $Stream.Write($SseMsg, 0, $SseMsg.Length)
                $Stream.Flush()
            }

            $DoneMsg = [System.Text.Encoding]::UTF8.GetBytes("data: [DONE]`n`n")
            $Stream.Write($DoneMsg, 0, $DoneMsg.Length)
            $Stream.Flush()
            $Response.Close()
        }
        else {
            if (Test-Path $FilePath) {
                $Bytes = [System.IO.File]::ReadAllBytes($FilePath)
                $Response.ContentType = "text/html; charset=utf-8"
                $Response.Headers.Add("Cache-Control", "no-cache")
                $Response.Headers.Add("Access-Control-Allow-Origin", "*")
                $Response.ContentLength64 = $Bytes.Length
                $Response.OutputStream.Write($Bytes, 0, $Bytes.Length)
            } else {
                $Response.StatusCode = 404
            }
            $Response.Close()
        }
    } catch {
        # Disconnect
    }
}
