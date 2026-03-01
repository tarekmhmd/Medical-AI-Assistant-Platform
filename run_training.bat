@echo off
cd /d "D:\project 2"
"D:\project 2\venv\Scripts\python.exe" "D:\project 2\glm5_train.py" --project_dir "D:/project 2" --mode resume --checkpoints_dir "D:/project 2/checkpoints/" --models "classification,segmentation" --dataset_dir "D:/project 2/data/" --save_every 1 --max_epochs 3 --batch_size 8 --report_file "D:/project 2/project_run_training_followup.json"
