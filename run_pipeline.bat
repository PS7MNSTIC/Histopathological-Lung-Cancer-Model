@echo off
echo ============================================================
echo  HAGCA-Net Pipeline Runner
echo ============================================================
SET PYTHON=C:\Users\bmsah\anaconda3\envs\lung_cancer\python.exe
SET SRC=C:\ml_project\src

echo.
echo [Step 2] Data Cleaning...
%PYTHON% %SRC%\01_data_cleaning.py
if errorlevel 1 (echo ERROR in Step 2 && pause && exit /b 1)

echo.
echo [Step 3] Data Splitting...
%PYTHON% %SRC%\02_data_splitting.py
if errorlevel 1 (echo ERROR in Step 3 && pause && exit /b 1)

echo.
echo [Step 4] Preprocessing...
%PYTHON% %SRC%\03_preprocessing.py
if errorlevel 1 (echo ERROR in Step 4 && pause && exit /b 1)

echo.
echo All pipeline steps complete!
pause
