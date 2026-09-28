param([Parameter(Mandatory=$true)][string]$Url,[Parameter(Mandatory=$true)][string]$Destination,[Parameter(Mandatory=$true)][long]$Size,[string]$Sha256,[int]$Parallel=4,[long]$ChunkSize=8388608)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$Parts=$Destination+'.parts'
$OutputPath=[IO.Path]::GetFullPath($Destination)
$PartsPath=[IO.Path]::GetFullPath($Parts)
if(-not [string]::Equals($PartsPath,$OutputPath+'.parts',[StringComparison]::OrdinalIgnoreCase)){
  throw 'Part directory is outside the intended download path.'
}
if(Test-Path -LiteralPath $PartsPath){
  $PartsItem=Get-Item -LiteralPath $PartsPath -Force
  if(($PartsItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){
    throw 'Part directory is a link; refusing to use it.'
  }
}
New-Item -ItemType Directory -Path $Parts -Force | Out-Null
$Count=[int][Math]::Ceiling($Size/[double]$ChunkSize)
0..($Count-1) | ForEach-Object -Parallel {
  $ErrorActionPreference='Stop'
  $ProgressPreference='SilentlyContinue'
  $Index=$_
  $Begin=[long]$Index*$using:ChunkSize
  $End=[Math]::Min($using:Size-1,$Begin+$using:ChunkSize-1)
  $PartFile=Join-Path $using:Parts ('{0:D5}.part' -f $Index)
  $Expected=$End-$Begin+1
  if((Test-Path -LiteralPath $PartFile) -and (Get-Item -LiteralPath $PartFile).Length -eq $Expected){return}
  for($Attempt=1;$Attempt -le 4;$Attempt++){
    try{
      $Response=Invoke-WebRequest -Uri $using:Url -Headers @{Range="bytes=$Begin-$End"} -OutFile $PartFile -PassThru -TimeoutSec 120
      if($Response.StatusCode -ne 206 -or (Get-Item -LiteralPath $PartFile).Length -ne $Expected){throw 'Invalid range response'}
      Write-Output "Part $($Index+1)/$using:Count ready"
      break
    }catch{
      if($Attempt -eq 4){throw}
      Start-Sleep -Seconds 2
    }
  }
} -ThrottleLimit $Parallel
$Output=[IO.File]::Create($Destination)
try{
  for($Index=0;$Index -lt $Count;$Index++){
    $PartFile=Join-Path $Parts ('{0:D5}.part' -f $Index)
    $Expected=[Math]::Min($ChunkSize,$Size-[long]$Index*$ChunkSize)
    if(!(Test-Path -LiteralPath $PartFile) -or (Get-Item -LiteralPath $PartFile).Length -ne $Expected){throw "Incomplete part: $Index"}
    $PartStream=[IO.File]::OpenRead($PartFile)
    try{$PartStream.CopyTo($Output)}finally{$PartStream.Dispose()}
  }
}finally{$Output.Dispose()}
$ActualHash=(Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
if($Sha256 -and $ActualHash -ne $Sha256){throw "SHA256 mismatch: $ActualHash"}
$PartsItem=Get-Item -LiteralPath $PartsPath -Force
if(($PartsItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){
  throw 'Part directory is a link; refusing to remove it.'
}
Remove-Item -LiteralPath $PartsPath -Recurse -Force
Write-Output "Complete: $Destination SHA256=$ActualHash"
