# AVALIAÇÃO OFICIAL (FASE F): CÓPIA DO Blocks/Marlim/Analysis.ipynb COM BASE_PATH/MODELS/PATCH_IDS TROCADOS, EXECUTADA POR PAPERMILL
# COM cwd=Blocks/Marlim. O NOTEBOOK DO USUÁRIO NÃO É TOCADO; A CÓPIA E A SAÍDA FICAM EM models/<run>/official/.
# uso: python official.py <pasta-com-model_N> <model_N> [patches separados por vírgula]
import os, sys, re, json, time
import nbformat
import papermill as pm

ROOT = '/home/grva-mint/Projects/Falhas'
SRC  = f'{ROOT}/Blocks/Marlim/Analysis.ipynb'


def patch(base, model, patches):
    swaps = [
        (r"^BASE_PATH  = .*$", f"BASE_PATH  = '{base}'"),
        (r"^MODELS    = .*$", f"MODELS    = ['{model}']"),
        (r"^PATCH_IDS = .*$", f"PATCH_IDS = {patches!r}"),
    ]
    nb    = nbformat.read(SRC, as_version=4)
    count = [0] * len(swaps)

    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        for k, (old, new) in enumerate(swaps):
            cell.source, n = re.subn(old, new, cell.source, flags=re.M)
            count[k] += n

    if count != [1] * len(swaps):
        raise ValueError(f'trocas que não casaram exatamente uma vez: {count}')

    return nb


def main(base, model, patches):
    out = f'{base}/{model}/official'
    os.makedirs(out, exist_ok=True)
    nbformat.write(patch(base, model, patches), f'{out}/Analysis.ipynb')
    t0 = time.time()
    pm.execute_notebook(f'{out}/Analysis.ipynb', f'{out}/Analysis_out.ipynb', kernel_name='python3', cwd=f'{ROOT}/Blocks/Marlim', progress_bar=False)
    print(model, patches, f'{(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(',') if len(sys.argv) > 3 else ['1200', '1300'])
