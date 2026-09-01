@echo off
:: micronaut-train.bat — direct dataset training for each MM-* micronaut slot
::
:: Pipeline per slot:
::   1. tokenize_transitions.py --chat --pack N  →  tokens\{slot}.bin
::   2. gpt2_trainer.exe --data .bin --completion-only  →  trained\{slot}.safetensors
::
:: No teacher / no distillation — pure dataset training on the GPU trainer.
:: --completion-only: loss only on <INSTRUCT>...</INSTRUCT> spans (not user prompts).
::
:: Usage:
::   micronaut-train [slot]      run one slot
::   micronaut-train all         run all slots sequentially
::   micronaut-train tok [slot]  tokenize only (skip trainer)
::   micronaut-train             show this help
setlocal EnableDelayedExpansion
set "ROOT=C:\Users\canna\_khanary_inspect"
set "PYTHON=python"
set "TOKENIZE=%ROOT%\tools\tokenize_transitions.py"
set "TRAINER=%ROOT%\trainer\build\Release\gpt2_trainer.exe"
set "BIN_DIR=%ROOT%\models\from_zero\tokens"
set "OUT_DIR=%ROOT%\models\from_zero\trained"
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

:: ── Data paths (read-only from E:\) ──────────────────────────────────────────
set "D_CODER=E:\data\data\coder_outputs\chunk_31.jsonl,E:\data\data\coder_outputs\chunk_32.jsonl,E:\data\data\coder_outputs\chunk_33.jsonl,E:\data\data\coder_outputs\chunk_34.jsonl,E:\data\data\coder_outputs\chunk_35.jsonl"
set "D_MATH=E:\data\data\chat-data\exam_ChatGLM3-6B.csv,E:\data\data\chat-data\exam_DeepSeek-66B.csv,E:\data\data\chat-data\exam_Qwen-14B-Chat.csv,E:\data\data\chat-data\exam_Yi-34B-Chat.csv,E:\data\data\chat-data\exam_Baichuan2-13B-Chat.csv"
set "D_TOOLCALL=E:\data\data\chat-data\opus46_final.jsonl"
set "D_KUHUL=%ROOT%\models\from_zero\rlhf_words.jsonl"
set "D_RESEARCH=E:\data\data\chat-data\deepseek67B.jsonl,E:\data\data\chat-data\deepseek-67b-1206-no-sp.jsonl,%ROOT%\models\from_zero\rlhf_words.jsonl"
set "D_AGENT=E:\data\data\chat-data\qwen14B.jsonl,E:\data\data\chat-data\yi34B.jsonl"
set "D_SKILL=E:\data\data\chat-data\baichuan13B.jsonl"
set "D_DESIGN=E:\data\data\chat-data\chatglm.jsonl"

:: ── Shared trainer params ─────────────────────────────────────────────────────
:: Model preset — distilgpt2 (82M) or gpt2-small (117M); set to safetensors path to resume
set "MODEL=distilgpt2"
set "STEPS=2000"
set "BATCH=4"
set "LR=3e-4"
set "BLOCK=128"
set "SAVE_EVERY=500"

if "%~1"==""          goto :usage
if /i "%~1"=="all"    goto :run_all
if /i "%~1"=="tok"    goto :tok_one
goto :run_one

:run_all
call :run_coder
call :run_slot mm-math     "%D_MATH%"
call :run_slot mm-toolcall "%D_TOOLCALL%"
call :run_slot mm-kuhul    "%D_KUHUL%"
call :run_slot mm-research "%D_RESEARCH%"
call :run_slot mm-agent    "%D_AGENT%"
call :run_slot mm-skill    "%D_SKILL%"
call :run_slot mm-design   "%D_DESIGN%"
echo.
echo [train] all slots done. Models in %OUT_DIR%\
goto :eof

:run_one
if /i "%~1"=="mm-coder"    call :run_coder & goto :eof
if /i "%~1"=="mm-math"     call :run_slot mm-math     "%D_MATH%"     & goto :eof
if /i "%~1"=="mm-toolcall" call :run_slot mm-toolcall "%D_TOOLCALL%" & goto :eof
if /i "%~1"=="mm-kuhul"    call :run_slot mm-kuhul    "%D_KUHUL%"    & goto :eof
if /i "%~1"=="mm-research" call :run_slot mm-research "%D_RESEARCH%" & goto :eof
if /i "%~1"=="mm-agent"    call :run_slot mm-agent    "%D_AGENT%"    & goto :eof
if /i "%~1"=="mm-skill"    call :run_slot mm-skill    "%D_SKILL%"    & goto :eof
if /i "%~1"=="mm-design"   call :run_slot mm-design   "%D_DESIGN%"   & goto :eof
echo [train] WARN: unknown slot '%~1'
goto :eof

:tok_one
if /i "%~2"=="mm-coder"    call :tokenize mm-coder    "%D_CODER%"    & goto :eof
if /i "%~2"=="mm-math"     call :tokenize mm-math     "%D_MATH%"     & goto :eof
if /i "%~2"=="mm-toolcall" call :tokenize mm-toolcall "%D_TOOLCALL%" & goto :eof
if /i "%~2"=="mm-kuhul"    call :tokenize mm-kuhul    "%D_KUHUL%"    & goto :eof
if /i "%~2"=="mm-research" call :tokenize mm-research "%D_RESEARCH%" & goto :eof
if /i "%~2"=="mm-agent"    call :tokenize mm-agent    "%D_AGENT%"    & goto :eof
if /i "%~2"=="mm-skill"    call :tokenize mm-skill    "%D_SKILL%"    & goto :eof
if /i "%~2"=="mm-design"   call :tokenize mm-design   "%D_DESIGN%"   & goto :eof
echo [train] WARN: unknown slot '%~2'
goto :eof

:run_coder
:: mm-coder: starts from ultrachat coder skeleton (10L 1024E gpt2-medium-ish).
:: --no-gpu-fwd: Phase 1 (CPU fwd/bwd + GPU Adam) — fits in 2 GB shared VRAM.
:: --group-by-lang + --no-shuffle: curriculum order (Python -> JS -> C++ -> ...).
set "_BIN=%BIN_DIR%\mm-coder.bin"
set "_OUT=%OUT_DIR%\mm-coder.safetensors"
set "_SKELETON=%ROOT%\models\from_zero\coder_skeleton.safetensors"
echo.
echo [train] === mm-coder -- tokenize (group-by-lang) ===
echo [train] src : %D_CODER%
echo [train] bin : %_BIN%
%PYTHON% "%TOKENIZE%" "%D_CODER%" "%_BIN%" --chat --pack %BLOCK% --group-by-lang
if not exist "%_BIN%" ( echo [train] ERROR: tokenize failed for mm-coder & goto :eof )
if not exist "%_SKELETON%" (
    echo [train] ERROR: coder skeleton not found: %_SKELETON%
    echo [train] Run: python tools\gguf_to_safetensors.py ^"E:\models\GPT2\coder_micronaut\ultrachat_coder_skeleton_slerp_0p40.gguf^" models\from_zero\coder_skeleton.safetensors --expand-vocab 50282
    goto :eof
)
echo.
echo [train] === mm-coder -- trainer (skeleton base, curriculum, Phase-1) ===
echo [train] base: %_SKELETON%
pushd "%ROOT%\trainer\build\Release"
"%TRAINER%" ^
    --model "%_SKELETON%" ^
    --data  "%_BIN%" ^
    --out   "%_OUT%" ^
    --steps  %STEPS% ^
    --batch  %BATCH% ^
    --lr     1e-5 ^
    --block  %BLOCK% ^
    --save-every %SAVE_EVERY% ^
    --completion-only ^
    --no-shuffle ^
    --no-gpu-fwd
popd
echo [train] mm-coder done -> %_OUT%
goto :eof

:run_slot
:: %1=slot  %2=comma-separated data paths
set "SLOT=%~1"
set "BIN=%BIN_DIR%\%SLOT%.bin"
set "OUT=%OUT_DIR%\%SLOT%.safetensors"

call :tokenize "%SLOT%" "%~2"
if not exist "%BIN%" ( echo [train] ERROR: tokenize failed for %SLOT% & goto :eof )

echo.
echo [train] === %SLOT% -- trainer ===
echo [train] data : %BIN%
echo [train] model: %MODEL%  steps: %STEPS%  batch: %BATCH%  lr: %LR%  block: %BLOCK%
echo.
pushd "%ROOT%\trainer\build\Release"
"%TRAINER%" ^
    --model "%MODEL%" ^
    --data  "%BIN%" ^
    --out   "%OUT%" ^
    --steps  %STEPS% ^
    --batch  %BATCH% ^
    --lr     %LR% ^
    --block  %BLOCK% ^
    --save-every %SAVE_EVERY% ^
    --completion-only
popd
echo [train] %SLOT% done -> %OUT%
goto :eof

:tokenize
:: %1=slot  %2=comma-separated data paths
set "_SLOT=%~1"
set "_DATA=%~2"
set "_BIN=%BIN_DIR%\%_SLOT%.bin"
echo.
echo [train] === %_SLOT% -- tokenize ===
echo [train] src : %_DATA%
echo [train] bin : %_BIN%
%PYTHON% "%TOKENIZE%" "%_DATA%" "%_BIN%" --chat --pack %BLOCK%
goto :eof

:usage
echo.
echo  Usage: micronaut-train [slot ^| all ^| tok slot]
echo.
echo  Slots:  mm-coder  mm-math  mm-toolcall  mm-kuhul  mm-research  mm-agent  mm-skill  mm-design
echo.
echo  Examples:
echo    micronaut-train mm-kuhul          tokenize + train mm-kuhul
echo    micronaut-train all               run all slots sequentially
echo    micronaut-train tok mm-coder      tokenize only, inspect .bin before training
echo.
echo  Params (edit top of this file):
echo    MODEL=%MODEL%   STEPS=%STEPS%   BATCH=%BATCH%   LR=%LR%   BLOCK=%BLOCK%
echo.
echo  Output:
echo    tokens → %BIN_DIR%\{slot}.bin
echo    models → %OUT_DIR%\{slot}.safetensors
echo.
endlocal
