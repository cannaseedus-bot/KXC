@echo off
:: micronaut-distil.bat — queue domain LoRA runs for each MM-* micronaut
::
:: Usage:
::   micronaut-distil [slot]        run one slot  (mm-coder, mm-math, mm-toolcall, …)
::   micronaut-distil all           run all slots sequentially
::   micronaut-distil               show this help
::
:: Each run trains a LoRA adapter on the from_zero student using pre-generated
:: completions from the corpus on E:\. No live teacher required.
::
:: Output LoRAs land in models\from_zero\loras\ named {slot}.safetensors.
:: Merge with tools\oss_distillation.py --resume to continue a run.
setlocal EnableDelayedExpansion
set "ROOT=C:\Users\canna\_khanary_inspect"
set "PYTHON=python"
set "SCRIPT=%ROOT%\tools\oss_distillation.py"
set "STUDENT=%ROOT%\models\from_zero\from_zero_v0.6_merged.gguf"
set "STUDENT_ST=%ROOT%\models\from_zero\from_zero_v0.6_merged.safetensors"
set "LORA_DIR=%ROOT%\models\from_zero\loras"
if not exist "%LORA_DIR%" mkdir "%LORA_DIR%"

:: ── Data paths (read-only from E:\) ──────────────────────────────────────────
set "D_CODER=E:\data\data\coder_outputs\chunk_31.jsonl,E:\data\data\coder_outputs\chunk_32.jsonl,E:\data\data\coder_outputs\chunk_33.jsonl,E:\data\data\coder_outputs\chunk_34.jsonl,E:\data\data\coder_outputs\chunk_35.jsonl"
set "D_MATH=E:\data\data\chat-data\exam_ChatGLM3-6B.csv,E:\data\data\chat-data\exam_DeepSeek-66B.csv,E:\data\data\chat-data\exam_Qwen-14B-Chat.csv,E:\data\data\chat-data\exam_Yi-34B-Chat.csv,E:\data\data\chat-data\exam_Baichuan2-13B-Chat.csv"
set "D_TOOLCALL=E:\data\data\chat-data\opus46_final.jsonl"
set "D_RESEARCH=E:\data\data\chat-data\deepseek67B.jsonl,E:\data\data\chat-data\deepseek-67b-1206-no-sp.jsonl,%ROOT%\models\from_zero\rlhf_words.jsonl"
set "D_AGENT=E:\data\data\chat-data\qwen14B.jsonl,E:\data\data\chat-data\yi34B.jsonl"
set "D_SKILL=E:\data\data\chat-data\baichuan13B.jsonl"
set "D_DESIGN=E:\data\data\chat-data\chatglm.jsonl"
set "D_KUHUL=%ROOT%\models\from_zero\rlhf_words.jsonl"
set "D_CHAT=E:\data\data\chat-data\opus46_final.jsonl,E:\data\data\chat-data\qwen14B.jsonl,E:\data\data\chat-data\yi34B.jsonl"

:: ── Shared training params ────────────────────────────────────────────────────
set "RANK=8"
set "STEPS=500"
set "LR=1e-4"
set "KUHUL=http://127.0.0.1:8764"
set "TEACHER=http://127.0.0.1:9003"

if "%~1"==""     goto :usage
if /i "%~1"=="all" goto :run_all
goto :run_one

:run_all
call :train mm-coder    "%D_CODER%"
call :train mm-math     "%D_MATH%"
call :train mm-toolcall "%D_TOOLCALL%"
call :train mm-kuhul    "%D_KUHUL%"
call :train mm-research "%D_RESEARCH%"
call :train mm-agent    "%D_AGENT%"
call :train mm-skill    "%D_SKILL%"
call :train mm-design   "%D_DESIGN%"
echo.
echo [distil] all slots done. LoRAs in %LORA_DIR%\
goto :eof

:run_one
if /i "%~1"=="mm-coder"    call :train mm-coder    "%D_CODER%"    & goto :eof
if /i "%~1"=="mm-math"     call :train mm-math     "%D_MATH%"     & goto :eof
if /i "%~1"=="mm-toolcall" call :train mm-toolcall "%D_TOOLCALL%" & goto :eof
if /i "%~1"=="mm-kuhul"    call :train_kuhul "%D_KUHUL%"            & goto :eof
if /i "%~1"=="mm-research" call :train mm-research "%D_RESEARCH%" & goto :eof
if /i "%~1"=="mm-agent"    call :train mm-agent    "%D_AGENT%"    & goto :eof
if /i "%~1"=="mm-skill"    call :train mm-skill    "%D_SKILL%"    & goto :eof
if /i "%~1"=="mm-design"   call :train mm-design   "%D_DESIGN%"   & goto :eof
echo  [distil] WARN: unknown slot '%~1'
goto :eof

:train_kuhul
:: mm-kuhul only — drains all paired data, no teacher (cloud has no kuhul knowledge).
set "OUT=%LORA_DIR%\mm-kuhul.safetensors"
echo.
echo [distil] === mm-kuhul (no-teacher, 1418 steps) ===
echo [distil] data : %~1
echo [distil] out  : %OUT%
echo.
%PYTHON% "%SCRIPT%" ^
    --student    "%STUDENT_ST%" ^
    --out        "%OUT%" ^
    --rank       %RANK% ^
    --steps      1418 ^
    --lr         %LR% ^
    --jsonl-data "%~1" ^
    --no-teacher ^
    --kuhul-server %KUHUL%
echo [distil] mm-kuhul done.
goto :eof

:train
:: %1=slot-name %2=data-paths
set "SLOT=%~1"
set "DATA=%~2"
set "OUT=%LORA_DIR%\%SLOT%.safetensors"
echo.
echo [distil] === %SLOT% ===
echo [distil] data : %DATA%
echo [distil] out  : %OUT%
echo [distil] steps: %STEPS%  rank: %RANK%  lr: %LR%
echo.
%PYTHON% "%SCRIPT%" ^
    --student    "%STUDENT_ST%" ^
    --out        "%OUT%" ^
    --rank       %RANK% ^
    --steps      %STEPS% ^
    --lr         %LR% ^
    --jsonl-data "%DATA%" ^
    --engine     %TEACHER% ^
    --kuhul-server %KUHUL%
echo [distil] %SLOT% done.
goto :eof

:usage
echo.
echo  Usage: micronaut-distil [slot ^| all]
echo.
echo  Slots:  mm-coder  mm-math  mm-toolcall  mm-kuhul  mm-research  mm-agent  mm-skill  mm-design
echo.
echo  Data mapping:
echo    mm-coder    -- coder_outputs chunk_31-35
echo    mm-math     -- exam CSVs (ChatGLM3/DeepSeek/Qwen/Yi/Baichuan)
echo    mm-toolcall -- opus46_final  (Claude Opus tool calling)
echo    mm-kuhul    -- rlhf_words  (stack intel, words-only)
echo    mm-research -- deepseek67B + deepseek-67b-1206-no-sp + rlhf_words
echo    mm-agent    -- qwen14B + yi34B
echo    mm-skill    -- baichuan13B
echo    mm-design   -- chatglm
echo.
endlocal
