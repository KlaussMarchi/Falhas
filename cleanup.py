import csv
import os
import shutil

csv_path = 'Searcher/Searcher/historico_otimizacao.csv'
backup_dir = 'Searcher/Model/Backup'

count_models = 0
freed_space = 0

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model_id = row['model_id']
        if model_id == 'N/A' or not model_id:
            continue
        try:
            iou_wu = float(row['iou_wu'])
        except ValueError:
            continue
            
        if iou_wu < 0.60:
            count_models += 1
            model_dir = os.path.join(backup_dir, model_id)
            if not os.path.exists(model_dir):
                continue
                
            model_pth = os.path.join(model_dir, 'model.pth')
            predictions_dir = os.path.join(model_dir, 'predictions')
            
            if os.path.exists(model_pth):
                size = os.path.getsize(model_pth)
                os.remove(model_pth)
                freed_space += size
                print(f"Removed {model_pth} ({(size/1024/1024):.2f} MB)")
                
            if os.path.exists(predictions_dir):
                # approximate size
                for dirpath, dirnames, filenames in os.walk(predictions_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            freed_space += os.path.getsize(fp)
                shutil.rmtree(predictions_dir)
                print(f"Removed directory {predictions_dir}")

print(f"\nCleanup complete. Freed approximately {(freed_space/1024/1024):.2f} MB")
