param(
    [Parameter(Mandatory = $true)]
    [string]$ConverterPath,
    [Parameter(Mandatory = $true)]
    [string]$OnnxPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPrefix
)

$ErrorActionPreference = 'Stop'
$converter = (Resolve-Path -LiteralPath $ConverterPath).Path
$model = (Resolve-Path -LiteralPath $OnnxPath).Path
& $converter --fmk=ONNX --modelFile=$model --outputFile=$OutputPrefix
if ($LASTEXITCODE -ne 0) {
    throw "MindSpore Lite conversion failed with exit code $LASTEXITCODE"
}
