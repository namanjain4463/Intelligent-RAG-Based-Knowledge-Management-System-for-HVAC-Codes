param(
    [string]$EnvFile = "",
    [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $ProjectRoot
if ($EnvFile) {
    $ResolvedEnv = (Resolve-Path -LiteralPath $EnvFile).Path
    $env:HVAC_ENV_FILE = $ResolvedEnv
}
# This launcher selects the validated v2 index; it does not rewrite a legacy .env.
$env:VECTOR_INDEX_NAME = "hvac_passage_embeddings"
$env:VECTOR_DIMENSION = "3072"
python -m streamlit run bot.py --server.port $Port
