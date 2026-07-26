<#
.SYNOPSIS
使用指定的 HiAI/CANN DDK 转换器和版本化参数文件生成离线模型。

.DESCRIPTION
脚本不内置可能随 DDK 版本变化的命令行参数。调用方必须把实际参数逐行写入
ArgumentsFile，并将转换器版本、参数文件和输出哈希记录到 model_manifest.json。
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ConverterPath,
    [Parameter(Mandatory = $true)]
    [string]$ArgumentsFile
)

$ErrorActionPreference = 'Stop'
# Resolve-Path 同时验证输入存在，并把审计日志中的路径统一为绝对路径。
$converter = (Resolve-Path -LiteralPath $ConverterPath).Path
$argumentsPath = (Resolve-Path -LiteralPath $ArgumentsFile).Path
# 空行与注释行不传给转换器，便于参数文件内保留中文设计说明。
$arguments = Get-Content -LiteralPath $argumentsPath |
    Where-Object { $_.Trim() -ne '' -and -not $_.Trim().StartsWith('#') }

Write-Host "Converter: $converter"
Write-Host "Arguments file: $argumentsPath"
Write-Host "The exact DDK version and this arguments file must be recorded in model_manifest.json."
& $converter @arguments
# PowerShell 对原生程序的非零退出码默认不会抛异常，因此必须显式检查。
if ($LASTEXITCODE -ne 0) {
    throw "HiAI converter failed with exit code $LASTEXITCODE"
}
