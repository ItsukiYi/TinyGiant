param(
    [Parameter(Mandatory = $true, HelpMessage = "源目录路径")]
    [string]$SourceDir,

    [Parameter(Mandatory = $true, HelpMessage = "目标目录路径")]
    [string]$DestDir,

    [Parameter(HelpMessage = "是否覆盖已存在的文件（默认跳过）")]
    [switch]$Force,

    [Parameter(HelpMessage = "展平到目标目录，不保留子目录结构")]
    [switch]$Flat
)

# 检查源目录是否存在
if (-not (Test-Path -Path $SourceDir -PathType Container)) {
    Write-Error "源目录不存在: $SourceDir"
    exit 1
}

# 创建目标目录（如果不存在）
if (-not (Test-Path -Path $DestDir -PathType Container)) {
    New-Item -Path $DestDir -ItemType Directory -Force | Out-Null
    Write-Host "已创建目标目录: $DestDir" -ForegroundColor Green
}

# 递归搜索所有 .txt 文件
$files = Get-ChildItem -Path $SourceDir -Filter "*.txt" -File -Recurse

if ($files.Count -eq 0) {
    Write-Host "未找到任何 .txt 文件。" -ForegroundColor Yellow
    exit 0
}

Write-Host "找到 $($files.Count) 个 .txt 文件，开始复制..." -ForegroundColor Cyan

$copied = 0
$skipped = 0

foreach ($file in $files) {
    if ($Flat) {
        # 展平模式：所有文件直接放到目标目录
        $destPath = Join-Path -Path $DestDir -ChildPath $file.Name
    }
    else {
        # 保持目录结构：计算相对于源目录的路径
        $relativePath = $file.DirectoryName.Substring($SourceDir.TrimEnd('\').Length)
        $targetSubDir = "$DestDir$relativePath"

        # 创建子目录（如果不存在）
        if (-not (Test-Path -Path $targetSubDir -PathType Container)) {
            New-Item -Path $targetSubDir -ItemType Directory -Force | Out-Null
        }

        $destPath = Join-Path -Path $targetSubDir -ChildPath $file.Name
    }

    if ((Test-Path -Path $destPath) -and -not $Force) {
        Write-Warning "跳过（已存在）: $($file.Name)"
        $skipped++
        continue
    }

    Copy-Item -Path $file.FullName -Destination $destPath -Force:$Force
    Write-Host "已复制: $($file.Name)" -ForegroundColor Green
    $copied++
}

Write-Host "`n完成！" -ForegroundColor Cyan
Write-Host "  复制成功: $copied" -ForegroundColor Green
if ($skipped -gt 0) {
    Write-Host "  跳过: $skipped" -ForegroundColor Yellow
}