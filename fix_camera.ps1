$deviceId = "USB\VID_0C45&PID_6D1A&MI_00\6&e201e23&0&0000"
Write-Host "Restarting Integrated Webcam driver..."
pnputil /restart-device $deviceId
Write-Host "Done. Testing camera..."
Start-Sleep -Seconds 3
Write-Host "Camera driver restart complete."
