#!/usr/bin/env powershell

$args | foreach {
  if ($_ -match "VS") {
    $use_msbuild = "TRUE"
  }
  else {
    $use_msbuild = "FALSE"
  }
}

#Remove-Item build -Force -Recurse -ErrorAction SilentlyContinue
New-Item -Path .\build_win -ItemType directory -Force
Set-Location build_win

$TC_FILE = "-DCMAKE_TOOLCHAIN_FILE=$env:WORKSPACE\vcpkg\scripts\buildsystems\vcpkg.cmake"
$VCPKG_TRIPLET = "-DVCPKG_TARGET_TRIPLET=$env:VCPKG_DEFAULT_TRIPLET"
if ( $use_msbuild )
{
  # VS
  cmake -G "Visual Studio 15 Win64" "-DCMAKE_BUILD_TYPE=Release" $TC_FILE $VCPKG_TRIPLET ..
  cmake --build . --config Release
}
else
{
  # Ninja
  cmake -G "Ninja" "-DCMAKE_BUILD_TYPE=Release" $TC_FILE $VCPKG_TRIPLET ..
  cmake --build . --config Release
}

Set-Location ..
