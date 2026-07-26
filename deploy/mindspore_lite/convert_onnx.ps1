<#
.SYNOPSIS
调用指定版本的 MindSpore Lite converter 把 ONNX 转换为离线模型。

.DESCRIPTION
该脚本只封装稳定的最小参数。正式转换前仍需按目标麒麟 9000 软件栈确认
converter 版本、目标设备选项和量化参数，并把完整命令记录到模型清单。
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$ConverterPath,
    [Parameter(Mandatory = $true)]
    [string]$OnnxPath,
    [Parameter(Mandatory = $true)]
    [string]$OutputPrefix
)

$ErrorActionPreference = 'Stop'
# 先解析并验证工具和 ONNX 文件，避免转换器用含糊的相对路径报错。
$converter = (Resolve-Path -LiteralPath $ConverterPath).Path
$model = (Resolve-Path -LiteralPath $OnnxPath).Path
& $converter --fmk=ONNX --modelFile=$model --outputFile=$OutputPrefix
# 原生进程失败不会自动遵循 ErrorActionPreference，需要手动转成异常。
if ($LASTEXITCODE -ne 0) {
    throw "MindSpore Lite conversion failed with exit code $LASTEXITCODE"
}
