@echo off
chcp 65001
cd /d "c:\Users\48520\Desktop\eoh多目标\examples\user_bp_online_OPRO\evaluation"
python convert_bpplib_manual.py --extracted_dir ../../../BPPLIB-main/extracted --output_dir bpplib_converted
pause

