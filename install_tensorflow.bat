@echo off
REM Install TensorFlow in the current Python environment
REM This script installs TensorFlow CPU version for compatibility

echo Installing TensorFlow...
echo.

REM Check if virtual environment is activated
python -c "import sys; print('Virtual env active' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'WARNING: No virtual environment detected')"
echo.

REM Install TensorFlow CPU version (faster, more compatible)
pip install tensorflow>=2.13.0,<2.18.0

echo.
echo Verifying TensorFlow installation...
python -c "import tensorflow as tf; print(f'SUCCESS: TensorFlow {tf.__version__} installed')"

echo.
echo Installation complete!
pause
