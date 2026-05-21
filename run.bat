@echo off
:: FAERS Pharmacovigilance Analytics Platform
:: Main runner script with menu

setlocal EnableDelayedExpansion

:main_menu
cls
echo ============================================================
echo       FAERS PHARMACOVIGILANCE ANALYTICS PLATFORM
echo ============================================================
echo.
echo  1. Install Technologies (Python, pip, Docker)
echo  2. Run EDA Pipeline (Clear outputs + Execute with progress bar)
echo  3. Run Data Cleaning Pipeline
echo  4. Run Hadoop MapReduce Pipeline
echo  5. Run Data Balance Pipeline
echo  6. Run All Pipelines
echo  7. Setup Virtual Environment
echo  8. Exit
echo.
echo ============================================================
set /p choice=Select option (1-8): 

if "%choice%"=="1" goto install_tech
if "%choice%"=="2" goto run_eda
if "%choice%"=="3" goto run_cleaning
if "%choice%"=="4" goto run_hadoop
if "%choice%"=="5" goto run_balance
if "%choice%"=="6" goto run_all
if "%choice%"=="7" goto setup_venv
if "%choice%"=="8" goto exit
goto main_menu

:install_tech
cls
echo ============================================================
echo   INSTALLING TECHNOLOGIES
echo ============================================================
echo.
echo [1/3] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.10+.
    pause
    goto main_menu
)
echo Python found:
python --version

echo.
echo [2/3] Installing pip packages...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
python -m pip install -r requirements_hadoop.txt --quiet
python -m pip install mrjob --quiet
echo Dependencies installed.

echo.
echo [3/3] Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker not found. Install from https://docker.com
) else (
    echo Docker found:
    docker --version
)
pause
goto main_menu

:run_eda
cls
echo ============================================================
echo   RUNNING EDA PIPELINE
echo ============================================================
echo.
echo [1/3] Clearing previous outputs...
if exist "outputs\eda_results" rd /s /q "outputs\eda_results"
if exist "outputs\mapreduce_results" rd /s /q "outputs\mapreduce_results"
mkdir "outputs\eda_results" 2>nul
mkdir "outputs\mapreduce_results" 2>nul

echo [2/3] Running EDA with progress bar...
echo.

set total=8
set count=0

for %%s in (01 02 03 04 05 06 07 08) do (
    set /a count+=1
    set bar=
    for /l %%i in (1,1,!count!) do set "bar=!bar!"
    for /l %%i in (!count!,1,8) do set "bar=!bar!"
    echo [!bar!] !count!/8 - Running %%s_*...
    python src/eda/00_eda_simple.py 2>nul
    goto :eda_done
)
:eda_done

echo.
echo [3/3] EDA completed! Results in outputs/eda_results/
pause
goto main_menu

:run_cleaning
cls
echo ============================================================
echo   RUNNING DATA CLEANING PIPELINE
echo ============================================================
echo.
echo [1/3] Clearing previous cleaning outputs...
if exist "data\clean" rd /s /q "data\clean"
mkdir "data\clean" 2>nul
mkdir "data\clean\outputs" 2>nul

echo [2/3] Running data cleaning pipeline...
echo.
python src/cleaning/cleaning_main.py

echo.
echo [3/3] Data cleaning completed! Cleaned data in data\clean\
pause
goto main_menu

:run_hadoop
cls
echo ============================================================
echo   RUNNING HADOOP MAPREDUCE PIPELINE
echo ============================================================
echo.
echo [INFO] Running Hadoop pipeline in local mode...
if exist "src\hadoop\mapper\run_faers_pipeline.py" (
    python src/hadoop/mapper/run_faers_pipeline.py --local
) else (
    echo [WARN] Hadoop scripts not available. Run from docker/ for full Hadoop.
)
echo.
echo [OK] Hadoop completed! Results in outputs/mapreduce_results/
pause
goto main_menu

:run_balance
cls
echo ============================================================
echo   RUNNING DATA BALANCE PIPELINE
echo ============================================================
echo.
echo [1/3] Checking dependencies (imbalanced-learn)...
python -m pip install imbalanced-learn --quiet
echo.
echo [2/3] Running data balance pipeline...
echo.
python src/balance/balance_main.py --all
echo.
echo [3/3] Data balance completed!
echo    Results in outputs/balance_results/
echo    CSV exports in outputs/balance_results/datasets/
echo    Summary in outputs/balance_results/balance_summary.json
pause
goto main_menu

:run_all
cls
echo ============================================================
echo   RUNNING ALL PIPELINES
echo ============================================================
echo.
echo Step 1: Running EDA...
python src/main.py --eda
echo.
echo Step 2: Running Data Cleaning...
python src/cleaning/cleaning_main.py
echo.
echo Step 3: Running Data Balance...
python src/balance/balance_main.py --all
echo.
echo Step 4: Running Hadoop...
python src/main.py --hadoop
echo.
echo [OK] All pipelines completed!
pause
goto main_menu

:setup_venv
cls
echo ============================================================
echo   SETTING UP VIRTUAL ENVIRONMENT
echo ============================================================
echo.
echo Creating virtual environment in .venv...
python -m venv .venv
if exist ".venv\Scripts\activate.bat" (
    echo Virtual environment created successfully!
    echo To activate: .venv\Scripts\activate
) else (
    echo Failed to create virtual environment.
)
pause
goto main_menu

:exit
cls
echo Thank you for using FAERS Analytics Platform!
exit /b 0
