@echo off
REM Install dependencies with compatible versions
C:\Users\dm888\anaconda3\python.exe -m pip install --upgrade pip setuptools wheel

REM Install NumPy 1.x explicitly
C:\Users\dm888\anaconda3\python.exe -m pip install "numpy>=1.23,<2.0"

REM Install other dependencies
C:\Users\dm888\anaconda3\python.exe -m pip install -r requirements.txt

REM Run streamlit
C:\Users\dm888\anaconda3\python.exe -m streamlit run dash.py

pause
