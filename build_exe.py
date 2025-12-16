import PyInstaller.__main__
import os
import shutil

# Clean previous builds
if os.path.exists('dist'):
    shutil.rmtree('dist')
if os.path.exists('build_temp'): # PyInstaller uses 'build' by default, but we have a folder named 'build'
    shutil.rmtree('build_temp')

print("Building Executable...")

# Determine separator based on OS
separator = ';' if os.name == 'nt' else ':'

PyInstaller.__main__.run([
    'build/gui.py',                       # Entry point
    '--name=SLT_Parking_System',          # Name of the exe
    '--onedir',                           # One directory (easier for debugging/assets)
    '--noconsole',                        # Hide console window
    '--clean',                            # Clean cache
    
    # Data Files (Source;Dest) or (Source:Dest)
    f'--add-data=build/assets{separator}build/assets',
    f'--add-data=best_ncnn_model{separator}best_ncnn_model',
    
    # Hidden Imports (Dependencies that PyInstaller might miss)
    '--hidden-import=ultralytics',
    '--hidden-import=easyocr',
    '--hidden-import=PIL',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    
    # Collect all data for complex packages
    '--collect-all=ultralytics',
    '--collect-all=easyocr',
    
    # Workaround for 'build' folder name conflict
    '--workpath=build_temp',
    '--specpath=.',
])

print("Build Complete! Check the 'dist/SLT_Parking_System' folder.")
