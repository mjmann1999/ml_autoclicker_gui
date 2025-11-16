# ML Autoclicker GUI  

This project is a machine‑learning inspired autoclicker built with Python and PyQt5. It allows you to automate mouse clicks in a way that mimics human behavior by using randomized intervals, durations, rest periods, and jitter.  

## Features  
- **Configurable click timing**: Set the mean and standard deviation for click intervals and durations. Optional dynamic standard deviation gain over time.  
- **Rest scheduling**: Specify first and subsequent rest intervals and durations with optional human‑like (log‑normal/gamma) distributions.  
- **Micro‑rests**: Enable micro‑rests that randomly pause clicking between events. Control the probability and duration range for these small breaks.  
- **Pause on user activity**: When the mouse moves faster than a threshold, the autoclicker automatically pauses for a configurable idle period.  
- **Area and jitter targeting**: Choose to click at the current cursor, within a jitter radius, or inside a defined box area. An area picker dialog helps you select coordinates for box mode.  
- **Profiles**: Save and load settings profiles to a JSON file so you can quickly switch between configurations.  
- **GUI interface**: A modern tabbed interface provides quick access to main controls, advanced settings, and profile management. Real‑time status updates show running/resting state, next rest countdown, and logs.  
- **Failsafe**: The `pyautogui.FAILSAFE` flag is enabled so moving the mouse to the upper‑left corner stops the autoclicker instantly.  

## Requirements  
- Python 3.8+  
- PyQt5  
- pyautogui  
- numpy  

Install the requirements using pip:  
```
pip install pyqt5 pyautogui numpy
```  

## Usage  
1. Run `python main.py` to launch the GUI.  
2. Adjust the click timing, rest schedule, and other settings in the **Advanced** tab.  
3. Optionally choose a target area or jitter radius under **Targeting**.  
4. Save your configuration as a profile in the **Profiles** tab if desired.  
5. Click **Start** on the **Main** tab to begin automatic clicking.  
6. Click **Stop** to halt the autoclicker. You can also trigger the failsafe by moving the mouse to the top‑left corner of your screen.  

## Disclaimer  
This tool is provided for educational and personal automation purposes. Use it responsibly and ensure that your automation complies with the terms of service of any software you interact with.
