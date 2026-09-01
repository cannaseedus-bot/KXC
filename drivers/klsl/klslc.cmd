@echo off
setlocal
set "ROOT=%~dp0"
"%ROOT%bin\klslc.exe" %*
exit /b %ERRORLEVEL%
