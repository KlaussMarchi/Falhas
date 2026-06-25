import json

with open('Model/PredMarlim.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    
    # 1. Update models list to include patches
    if "models = [path for path in os.listdir(BASE_PATH)]" in source:
        new_source = source.replace("models = [path for path in os.listdir(BASE_PATH)]", "models = [path for path in os.listdir(BASE_PATH)]\npatches_to_predict = ['patch_1200']")
        new_source = new_source.replace("models", "models, patches_to_predict")
        cell['source'] = [line + '\n' for line in new_source.split('\n')]
        cell['source'][-1] = cell['source'][-1].strip() # last line no newline
        
    # 2. Remove standalone df creation
    if "df = pd.DataFrame({" in source and "'img_path': getFiles" in source:
        cell['source'] = []
        
    # 3. Remove standalone PredictDataset creation
    if "predictDataset = PredictDataset(df)" in source:
        cell['source'] = []
        
    # 4. Modify the main loop
    if "for model_name in models:" in source:
        new_loop = """SAVE_IMAGES = False

for model_name in models:
    model_path = f'{BASE_PATH}/{model_name}' 
    
    with open(f'{model_path}/info.json', 'r', encoding='utf-8') as f:
        modelInfo = json.load(f)

    modelOptions = modelInfo.get('model', {})
    print(f"\\n[{model_name}] Model Options:")
    print(json.dumps(modelOptions, indent=4))

    network   = ModelNetwork(**modelOptions)
    modelData = torch.load(f'{model_path}/model.pth')

    network.model.load_state_dict(modelData['model'])
    network.model.eval()
    network.model.to(network.device)
    print(f"Weights from {model_path} loaded successfully!\\n")

    for patch_id in patches_to_predict:
        # Cria dataframe e DataLoader para o patch
        img_paths = getFiles(f'../Dataset/marlim/{patch_id}/images')
        df = pd.DataFrame({'img_path': img_paths, 'shape': '(128, 128, 128)'})
        
        predictDataset = PredictDataset(df)
        predictLoader  = DataLoader(
            predictDataset, 
            batch_size=1,
            shuffle=False, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        mask_dat_dir  = os.path.join(model_path, 'marlim', patch_id, 'masks')
        os.makedirs(mask_dat_dir, exist_ok=True)

        print(f"Iniciando predições para {model_name} - {patch_id}...")

        with torch.no_grad():
            for index, img_tensor in enumerate(tqdm(predictLoader)):
                orig_path = df.iloc[index]['img_path']
                base_name = os.path.basename(orig_path).replace('.dat', '')
                filename_dat = f"{base_name}.dat"

                img_batch = img_tensor.to(network.device)
                logits = network.model(img_batch)

                if hasattr(network, 'multiclass') and network.multiclass:
                    probs = torch.softmax(logits, dim=1)
                else:
                    probs = torch.sigmoid(logits)

                probs_np = probs.squeeze().cpu().numpy().astype(np.float32)
                probs_np.tofile(os.path.join(mask_dat_dir, filename_dat))

        print(f"Predições para {model_name} - {patch_id} salvas com sucesso!\\n")
"""
        cell['source'] = [line + '\n' for line in new_loop.split('\n')[:-1]]

with open('Model/PredMarlim.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)
