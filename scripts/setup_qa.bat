@echo off
REM Script para configurar ferramentas de qualidade no Windows

echo 🔧 Configurando ferramentas de qualidade Python...

REM Ativar ambiente virtual se existir
if exist "venv\Scripts\activate.bat" (
    echo 📦 Ativando ambiente virtual...
    call venv\Scripts\activate.bat
)

REM Instalar dependências de desenvolvimento
echo 📥 Instalando ferramentas de QA...
pip install -r requirements-dev.txt

echo.
echo ✅ Configuração concluída!
echo.
echo 🧪 Para testar, execute:
echo   python -m pytest tests/ -v
echo   python -m flake8 src/
echo   python -m black src/ --check
echo   python -m mypy src/
echo.
echo 🎨 Para formatar código automaticamente:
echo   python -m black src/
echo.