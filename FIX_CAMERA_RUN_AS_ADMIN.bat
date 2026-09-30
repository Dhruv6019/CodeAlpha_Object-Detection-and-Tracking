@echo off
echo ============================================
echo   Integrated Webcam Driver Reset Tool
echo ============================================
echo.
echo This will disable and re-enable your webcam driver.
echo Requires Administrator privileges.
echo.

:: Disable webcam
echo [1/3] Disabling webcam driver...
pnputil /disable-device "USB\VID_0C45&PID_6D1A&MI_00\6&e201e23&0&0000"
timeout /t 2 /nobreak >nul

:: Enable webcam  
echo [2/3] Re-enabling webcam driver...
pnputil /enable-device "USB\VID_0C45&PID_6D1A&MI_00\6&e201e23&0&0000"
timeout /t 3 /nobreak >nul

echo [3/3] Done! Testing camera now...
python -c "import cv2,numpy,time; cap=cv2.VideoCapture(0,cv2.CAP_DSHOW); cap.set(cv2.CAP_PROP_BUFFERSIZE,1); time.sleep(2); [cap.read() for _ in range(10)]; ret,frame=cap.read(); print('Camera OK! Brightness='+str(round(float(numpy.mean(frame)),1)) if ret and frame is not None and float(numpy.mean(frame))>5 else 'Still black - try restarting PC'); cap.release()"
echo.
echo Now run: python main.py --source 0 --mirror
pause
