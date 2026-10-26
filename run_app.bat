@echo off
echo ======================================================================
echo  KAISER PERMANENTE MEMBER EXPERIENCE INTELLIGENCE PLATFORM
echo  Launching Executive Streamlit Hub & Care Navigation AI...
echo ======================================================================

set "PATH=%APPDATA%\Python\Python313\Scripts;C:\Python313\Scripts;%PATH%"

python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Streamlit exited with error code %ERRORLEVEL%.
    pause
)
