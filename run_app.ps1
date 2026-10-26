# ======================================================================
# KAISER PERMANENTE MEMBER EXPERIENCE INTELLIGENCE PLATFORM
# Launching Executive Streamlit Hub & Care Navigation AI (PowerShell)
# ======================================================================

$env:PATH = "$env:APPDATA\Python\Python313\Scripts;C:\Python313\Scripts;$env:PATH"

Write-Host "Starting Kaiser Permanente Streamlit Hub..." -ForegroundColor Cyan
python -m streamlit run app.py
