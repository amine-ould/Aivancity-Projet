@REM ============================================================
@REM WANDB Setup Script for Windows
@REM Usage: setup_wandb.bat
@REM ============================================================

@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   WANDB SETUP - Windows
echo ============================================================
echo.

REM Check Python
echo [1/4] Verification de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python n'est pas installe ou non accessible
    pause
    exit /b 1
)
echo OK: Python trouve
echo.

REM Install wandb
echo [2/4] Installation de WandB...
pip install wandb >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Impossible d'installer wandb
    pause
    exit /b 1
)
echo OK: WandB installe
echo.

REM Check wandb
echo [3/4] Verification de l'installation...
python -c "import wandb; print(f'WandB {wandb.__version__} OK')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WandB non verifie
    pause
    exit /b 1
)
echo OK: WandB verifie
echo.

REM Login
echo [4/4] Authentification WandB...
echo Allez sur: https://wandb.ai/authorize
echo Copiez votre API Key
echo.
set /p WANDB_KEY="Entrez votre API Key: "
if "!WANDB_KEY!"=="" (
    echo WARNING: Pas d'API Key fournie
    echo Vous pouvez vous authentifier plus tard avec: wandb login
) else (
    setx WANDB_API_KEY !WANDB_KEY!
    echo OK: API Key definie
)

echo.
echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo.
echo Prochaines etapes:
echo 1. Editer WANDB_CONFIG dans run_training.py (optionnel)
echo 2. Lancer: python run_training.py
echo 3. Ouvrir le lien WandB affiche
echo.

pause
