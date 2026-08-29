# terminal.ps1 - Direct, Untampered Conversational Dual Terminal
$Host.UI.RawUI.WindowTitle = "Cordis-OxCaml Dual Inference Terminal"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = $PSScriptRoot
$ModelPath = Join-Path $Root "models\Puro-2B-Base.Q4_K_M.gguf"
$LlamaCli = "C:\Users\asd\.gemini\antigravity\brain\6253f168-eef2-4c57-8a86-34f7be702a2a\scratch\llamacpp_bin\llama-cli.exe"
$LlamaBench = "C:\Users\asd\.gemini\antigravity\brain\6253f168-eef2-4c57-8a86-34f7be702a2a\scratch\llamacpp_bin\llama-bench.exe"
$LlamamlExe = Join-Path $Root "llamaml\_build\default\bin\main.exe"

$CurrentMode = "llamacpp"  # "llamacpp" or "llamaml"

function Show-Header {
    Clear-Host
    Write-Host "================================================================================" -ForegroundColor Magenta
    Write-Host "  CORDIS-OXCAML DUAL-ENGINE INTERACTIVE CONVERSATION TERMINAL                   " -ForegroundColor White
    Write-Host "  Model: Puro-2B-Base.Q4_K_M (2.03B params, 1.19 GiB)                           " -ForegroundColor DarkGray
    Write-Host "================================================================================" -ForegroundColor Magenta
    
    # Engine Selector Bar
    Write-Host " ACTIVE ENGINE: " -NoNewline -ForegroundColor DarkGray
    if ($CurrentMode -eq "llamacpp") {
        Write-Host " [1] LLAMA.CPP (C++/AVX2) [ACTIVE] " -NoNewline -BackgroundColor DarkBlue -ForegroundColor White
        Write-Host "   " -NoNewline
        Write-Host " [2] LLAMAML (Cordis-OxCaml 5+) " -ForegroundColor DarkGray
    } else {
        Write-Host " [1] LLAMA.CPP (C++/AVX2) " -NoNewline -ForegroundColor DarkGray
        Write-Host "   " -NoNewline
        Write-Host " [2] LLAMAML (Cordis-OxCaml 5+) [ACTIVE] " -BackgroundColor DarkMagenta -ForegroundColor White
    }
    
    Write-Host "`n Quick Controls: " -NoNewline -ForegroundColor DarkGray
    Write-Host "[1] " -NoNewline -ForegroundColor Cyan
    Write-Host "llama.cpp | " -NoNewline -ForegroundColor Gray
    Write-Host "[2] " -NoNewline -ForegroundColor Magenta
    Write-Host "Llamaml | " -NoNewline -ForegroundColor Gray
    Write-Host "[3] " -NoNewline -ForegroundColor Yellow
    Write-Host "Benchmark | " -NoNewline -ForegroundColor Gray
    Write-Host "[4] " -NoNewline -ForegroundColor Green
    Write-Host "Web UI | " -NoNewline -ForegroundColor Gray
    Write-Host "[Q] " -NoNewline -ForegroundColor Red
    Write-Host "Exit" -ForegroundColor Gray
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
}

function Run-Inference($prompt) {
    if ([string]::IsNullOrWhiteSpace($prompt)) { return }

    Write-Host "`n>>> Prompt: " -NoNewline -ForegroundColor Cyan
    Write-Host $prompt -ForegroundColor White

    if ($CurrentMode -eq "llamacpp") {
        Write-Host "`n[Direct llama.cpp C++ Engine Output]:" -ForegroundColor Blue
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
        
        # Direct, untampered execution of llama-cli
        $formatted = "Question: $prompt`nAnswer:"
        & $LlamaCli -m $ModelPath -p $formatted -n 64 --temp 0.7 --top-p 0.9 -t 4 --single-turn
        
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
    } 
    else {
        Write-Host "`n[Direct Llamaml OxCaml 5+ Engine Output]:" -ForegroundColor Magenta
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
        
        if (Test-Path $LlamamlExe) {
            # Direct, untampered execution of compiled Llamaml binary
            $formatted = "Question: $prompt`nAnswer:"
            & $LlamamlExe run --model $ModelPath --prompt $formatted --max-tokens 64 --temp 0.7 --top-p 0.9
        } else {
            Write-Host "Compiling Llamaml native binary..." -ForegroundColor DarkGray
            $env:PATH = "C:\Users\asd\AppData\Local\opam\5.2.1\bin;C:\Users\asd\AppData\Local\opam\.cygwin\root\usr\x86_64-w64-mingw32\sys-root\mingw\bin;" + $env:PATH
            Push-Location "$Root\llamaml"
            dune build bin/main.exe
            Pop-Location
            $formatted = "Question: $prompt`nAnswer:"
            & $LlamamlExe run --model $ModelPath --prompt $formatted --max-tokens 64 --temp 0.7 --top-p 0.9
        }
        
        Write-Host "--------------------------------------------------------------------------------" -ForegroundColor DarkGray
    }
}

function Run-Benchmark {
    Write-Host "`n[Executing Head-to-Head Hardware Benchmark (4 Threads)]..." -ForegroundColor Yellow
    & $LlamaBench -m $ModelPath -p 16 -n 32 -t 4
    Write-Host "`nBenchmark Complete. Press any key to continue..." -ForegroundColor Gray
    [Console]::ReadKey($true) | Out-Null
}

function Open-Dashboard {
    Write-Host "`nStarting native Llamaml Web Dashboard on http://localhost:8092..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command `$host.UI.RawUI.WindowTitle = 'LlamamlDaemon'; & '$Root\llamaml\server.ps1'" -WindowStyle Hidden
    Start-Process "http://localhost:8092"
    Start-Sleep -Seconds 1
}

# Main Interactive Loop
Show-Header

while ($true) {
    Write-Host "`nEnter prompt or command [1-4, Q]: " -NoNewline -ForegroundColor Yellow
    $inputStr = Read-Host

    if ($inputStr -eq "1") {
        $CurrentMode = "llamacpp"
        Show-Header
        Write-Host "`nSwitched to [1] LLAMA.CPP (C++/AVX2 Engine)" -ForegroundColor Blue
    }
    elseif ($inputStr -eq "2") {
        $CurrentMode = "llamaml"
        Show-Header
        Write-Host "`nSwitched to [2] LLAMAML (Cordis-OxCaml 5+ Algebraic Effects Engine)" -ForegroundColor Magenta
    }
    elseif ($inputStr -eq "3") {
        Run-Benchmark
        Show-Header
    }
    elseif ($inputStr -eq "4") {
        Open-Dashboard
        Show-Header
    }
    elseif ($inputStr -eq "q" -or $inputStr -eq "exit") {
        Write-Host "`nExiting Cordis-OxCaml Terminal. Goodbye!" -ForegroundColor DarkGray
        break
    }
    else {
        Run-Inference $inputStr
    }
}
