@echo off
cd /d G:\akhtar_khan
aktts\Scripts\python.exe scripts\finetune_tts.py --epochs 100 --patience 10 >> logs\finetune_stdout.log 2>> logs\finetune_stderr.log
