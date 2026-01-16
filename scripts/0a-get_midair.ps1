# Download files listed in download_config.txt
Get-Content $DownloadConfig | ForEach-Object {

    if (-not $_ -or $_.StartsWith("#")) {
        return
    }

    $url = $_.Trim()

    # Extract the file name from the URL and remove query parameters
    $uri = [System.Uri]$url
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($uri.AbsolutePath) + [System.IO.Path]::GetExtension($uri.AbsolutePath)
    $outPath = Join-Path $DatasetDir $filename

    Write-Host "Downloading $filename from $url"
    Invoke-WebRequest -Uri $url -OutFile $outPath
}
