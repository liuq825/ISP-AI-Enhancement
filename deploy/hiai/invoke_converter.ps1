param(
    [Parameter(Mandatory = $true)]
    [string]$ConverterPath,
    [Parameter(Mandatory = $true)]
    [string]$ArgumentsFile
)

$ErrorActionPreference = 'Stop'
$converter = (Resolve-Path -LiteralPath $ConverterPath).Path
$argumentsPath = (Resolve-Path -LiteralPath $ArgumentsFile).Path
$arguments = Get-Content -LiteralPath $argumentsPath |
    Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') }

Write-Host "Converter: $converter"
Write-Host "Arguments file: $argumentsPath"
Write-Host "The exact DDK version and this arguments file must be recorded in model_manifest.json."
& $converter @arguments
if ($LASTEXITCODE -ne 0) {
    throw "HiAI converter failed with exit code $LASTEXITCODE"
}
