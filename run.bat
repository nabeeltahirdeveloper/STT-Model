@echo off
REM ---------------------------------------------------------------------------
REM  Roman Urdu Auto-Captioning - task runner (Windows)
REM  Usage: run.bat <command> [args]
REM         run.bat help
REM ---------------------------------------------------------------------------
setlocal EnableDelayedExpansion

set "PROJECT_NAME=roman-urdu-captions"
set "PYTHON_VERSION=3.12"
set "VENV_DIR=.venv"
set "SRC_DIRS=src tests scripts"

REM ------------------------------- colours -----------------------------------
REM Enable ANSI on Windows 10+ consoles
for /f "tokens=2 delims=[]" %%A in ('ver') do set "WINVER=%%A"
reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1

if defined NO_COLOR (
    set "C_RESET=" & set "C_BOLD=" & set "C_DIM="
    set "C_RED="   & set "C_GREEN=" & set "C_YELLOW="
    set "C_BLUE="  & set "C_MAGENTA=" & set "C_CYAN="
) else (
    for /f %%E in ('echo prompt $E ^| cmd') do set "ESC=%%E"
    set "C_RESET=!ESC![0m"   & set "C_BOLD=!ESC![1m"     & set "C_DIM=!ESC![2m"
    set "C_RED=!ESC![0;31m"  & set "C_GREEN=!ESC![0;32m" & set "C_YELLOW=!ESC![0;33m"
    set "C_BLUE=!ESC![0;34m" & set "C_MAGENTA=!ESC![0;35m" & set "C_CYAN=!ESC![0;36m"
)

set "HR=%C_DIM%--------------------------------------------------------------------%C_RESET%"

REM ------------------------------- dispatch ----------------------------------
set "CMD=%~1"
if "%CMD%"=="" set "CMD=help"
shift

if /i "%CMD%"=="setup"       goto :cmd_setup
if /i "%CMD%"=="doctor"      goto :cmd_doctor
if /i "%CMD%"=="lint"        goto :cmd_lint
if /i "%CMD%"=="format"      goto :cmd_format
if /i "%CMD%"=="fmt"         goto :cmd_format
if /i "%CMD%"=="typecheck"   goto :cmd_typecheck
if /i "%CMD%"=="test"        goto :cmd_test
if /i "%CMD%"=="check"       goto :cmd_check
if /i "%CMD%"=="lexicon"     goto :cmd_lexicon
if /i "%CMD%"=="transcribe"  goto :cmd_transcribe
if /i "%CMD%"=="train"       goto :cmd_train
if /i "%CMD%"=="eval"        goto :cmd_eval
if /i "%CMD%"=="serve"       goto :cmd_serve
if /i "%CMD%"=="clean"       goto :cmd_clean
if /i "%CMD%"=="help"        goto :cmd_help
if /i "%CMD%"=="-h"          goto :cmd_help
if /i "%CMD%"=="--help"      goto :cmd_help

call :err unknown command: %CMD%
echo.
goto :cmd_help

REM ------------------------------ log helpers --------------------------------
:info
echo %C_BLUE%[ INFO ]%C_RESET% %*
exit /b 0

:ok
echo %C_GREEN%[  OK  ]%C_RESET% %*
exit /b 0

:warn
echo %C_YELLOW%[ WARN ]%C_RESET% %* 1>&2
exit /b 0

:err
echo %C_RED%[ FAIL ]%C_RESET% %* 1>&2
exit /b 0

:step
echo %C_MAGENTA%[ STEP ]%C_RESET% %C_BOLD%%*%C_RESET%
exit /b 0

:runcmd
echo %C_CYAN%[  -^>  ]%C_RESET% %C_DIM%%*%C_RESET%
call %*
if errorlevel 1 (
    call :err command failed: %*
    exit /b 1
)
exit /b 0

:banner
echo %HR%
echo %C_BOLD%%C_CYAN%  %PROJECT_NAME%%C_RESET%
echo %HR%
exit /b 0

:activate
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
) else (
    call :warn no virtualenv at %VENV_DIR% - run "run.bat setup" first
)
exit /b 0

REM ------------------------------- commands ----------------------------------
:cmd_setup
call :banner
call :step Environment setup

where python >nul 2>&1
if errorlevel 1 ( call :err python not found on PATH & exit /b 1 )
for /f "tokens=2" %%V in ('python --version 2^>^&1') do set "PYVER=%%V"
call :info python !PYVER! detected ^(target %PYTHON_VERSION%^)

where git >nul 2>&1
if errorlevel 1 ( call :err git not found on PATH & exit /b 1 )

where ffmpeg >nul 2>&1
if errorlevel 1 (
    call :warn ffmpeg NOT found - required for audio extraction
    call :info   install: winget install Gyan.FFmpeg
) else (
    call :ok ffmpeg found
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :warn no NVIDIA GPU - training will not be possible locally
) else (
    call :ok NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

call :step Creating virtualenv
if exist "%VENV_DIR%" (
    call :info reusing existing %VENV_DIR%
) else (
    call :runcmd python -m venv %VENV_DIR%
    if errorlevel 1 exit /b 1
)
call :activate

call :step Installing dependencies
call :runcmd python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
call :runcmd python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1

call :step Creating directory tree
for %%D in (data\raw data\labels data\eval data\lexicon data\work) do (
    if not exist "%%D" mkdir "%%D"
)
for %%D in (ingest preprocess labeling training inference subtitle api) do (
    if not exist "src\%%D" mkdir "src\%%D"
)
for %%D in (tests scripts notebooks configs docs) do (
    if not exist "%%D" mkdir "%%D"
)
call :ok directories created

if not exist ".git" (
    call :step Initialising git
    call :runcmd git init -q
)

where pre-commit >nul 2>&1
if not errorlevel 1 call :runcmd pre-commit install

echo %HR%
call :ok setup complete
call :info next: %C_BOLD%run.bat check%C_RESET%
exit /b 0

:cmd_lint
call :activate
call :step Lint ^(ruff^)
call :runcmd ruff check %SRC_DIRS%
if errorlevel 1 exit /b 1
call :ok lint passed
exit /b 0

:cmd_format
call :activate
call :step Format ^(ruff^)
call :runcmd ruff format %SRC_DIRS%
call :runcmd ruff check --fix %SRC_DIRS%
call :ok formatted
exit /b 0

:cmd_typecheck
call :activate
call :step Type check ^(mypy^)
call :runcmd mypy src
if errorlevel 1 exit /b 1
call :ok types passed
exit /b 0

:cmd_test
call :activate
call :step Tests ^(pytest^)
call :runcmd pytest -q --cov=src --cov-report=term-missing %*
if errorlevel 1 exit /b 1
call :ok tests passed
exit /b 0

:cmd_check
call :banner
call :info running all quality gates
call :cmd_lint
if errorlevel 1 exit /b 1
call :cmd_typecheck
if errorlevel 1 exit /b 1
call :cmd_test
if errorlevel 1 exit /b 1
echo %HR%
call :ok %C_BOLD%all gates green%C_RESET%
exit /b 0

:cmd_lexicon
call :activate
call :step Building frequency lexicon from Roman-Urdu-Parl
if not exist "data\raw\roman-urdu-parl" (
    call :err corpus missing: data\raw\roman-urdu-parl
    exit /b 1
)
call :runcmd python -m scripts.build_lexicon --corpus data/raw/roman-urdu-parl --top-n 5000 --out docs/lexicon-frequency.tsv
if errorlevel 1 exit /b 1
call :ok lexicon written to docs\lexicon-frequency.tsv
exit /b 0

:cmd_transcribe
call :activate
if "%~1"=="" (
    call :err usage: run.bat transcribe ^<video-file^> [--out DIR]
    exit /b 1
)
call :step Transcribing: %~1
call :runcmd python -m src.api.pipeline --input %*
if errorlevel 1 exit /b 1
call :ok done
exit /b 0

:cmd_train
call :activate
set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=configs/phase1.yaml"
if not exist "%CONFIG%" ( call :err config not found: %CONFIG% & exit /b 1 )
call :step Training with %CONFIG%
call :warn verify data\eval\ is NOT in the training manifest before proceeding
call :runcmd python -m src.training.finetune --config %CONFIG%
if errorlevel 1 exit /b 1
call :ok training complete
exit /b 0

:cmd_eval
call :activate
set "PRED=%~1"
set "REF=%~2"
if "%PRED%"=="" set "PRED=out/predictions.txt"
if "%REF%"==""  set "REF=data/eval/reference.txt"
call :step Evaluating
call :runcmd python -m src.eval.score --pred %PRED% --ref %REF% --metrics cer,sn-wer,normalized-wer,english-preservation
if errorlevel 1 exit /b 1
call :ok evaluation complete
exit /b 0

:cmd_serve
call :activate
if not defined MODEL set "MODEL=Qwen/Qwen3-ASR-1.7B"
if not defined PORT  set "PORT=8000"
call :step Serving %MODEL% on port %PORT%
call :runcmd qwen-asr-serve %MODEL% --gpu-memory-utilization 0.8 --host 0.0.0.0 --port %PORT%
exit /b 0

:cmd_clean
call :step Cleaning build artefacts
for /d /r . %%D in (__pycache__) do @if exist "%%D" rd /s /q "%%D"
for %%D in (.pytest_cache .mypy_cache .ruff_cache htmlcov dist build) do (
    if exist "%%D" rd /s /q "%%D"
)
if exist ".coverage" del /q ".coverage"
call :ok cleaned ^(data\ and .venv\ untouched^)
exit /b 0

:cmd_doctor
call :banner
call :step Diagnostics
set /a ISSUES=0
call :_chk "python"            "where python"
call :_chk "ffmpeg"            "where ffmpeg"
call :_chk "git"               "where git"
call :_chk "NVIDIA GPU"        "where nvidia-smi"
if exist "%VENV_DIR%" (call :ok virtualenv) else (call :warn virtualenv - MISSING & set /a ISSUES+=1)
if exist "data\eval\reference.txt" (call :ok eval set) else (call :warn eval set - MISSING & set /a ISSUES+=1)
if exist "data\lexicon\canonical.tsv" (call :ok canonical lexicon) else (call :warn canonical lexicon - MISSING & set /a ISSUES+=1)
if exist "data\lexicon\english.txt" (call :ok english lexicon) else (call :warn english lexicon - MISSING & set /a ISSUES+=1)
echo %HR%
if !ISSUES!==0 ( call :ok no issues ) else ( call :warn !ISSUES! issue^(s^) found )
exit /b 0

:_chk
%~2 >nul 2>&1
if errorlevel 1 (
    call :warn %~1 - MISSING
    set /a ISSUES+=1
) else (
    call :ok %~1
)
exit /b 0

:cmd_help
call :banner
echo %C_BOLD%USAGE%C_RESET%
echo   run.bat ^<command^> [args]
echo.
echo %C_BOLD%SETUP%C_RESET%
echo   %C_GREEN%setup%C_RESET%        Create venv, install deps, scaffold directories
echo   %C_GREEN%doctor%C_RESET%       Diagnose missing tools and data
echo.
echo %C_BOLD%QUALITY%C_RESET%
echo   %C_GREEN%check%C_RESET%        Run lint + typecheck + test ^(use before every commit^)
echo   %C_GREEN%lint%C_RESET%         ruff check
echo   %C_GREEN%format%C_RESET%       ruff format + autofix
echo   %C_GREEN%typecheck%C_RESET%    mypy
echo   %C_GREEN%test%C_RESET%         pytest with coverage
echo.
echo %C_BOLD%PIPELINE%C_RESET%
echo   %C_GREEN%lexicon%C_RESET%      Build frequency lexicon from Roman-Urdu-Parl
echo   %C_GREEN%transcribe%C_RESET%   Transcribe a video file
echo   %C_GREEN%train%C_RESET%        Fine-tune  ^(default configs/phase1.yaml^)
echo   %C_GREEN%eval%C_RESET%         Score predictions ^(CER, SN-WER, ...^)
echo   %C_GREEN%serve%C_RESET%        Start vLLM inference server
echo.
echo %C_BOLD%MISC%C_RESET%
echo   %C_GREEN%clean%C_RESET%        Remove caches and build artefacts
echo   %C_GREEN%help%C_RESET%         This message
echo.
echo %C_BOLD%ENVIRONMENT%C_RESET%
echo   MODEL=^<hf-id^>   override serving model
echo   PORT=^<n^>        override serving port
echo   NO_COLOR=1      disable coloured output
exit /b 0
