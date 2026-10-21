# EVIDÊNCIA DE REJEITO SEM RÓTULO: AO LONGO DE UM STICK, O TRAÇO COMPOSTO DE CADA LADO (SEGUINDO O MERGULHO DAS CAMADAS ATÉ A
# FALHA) É CORRELACIONADO EM DESLOCAMENTO VERTICAL; FALHA = CORRELAÇÃO ALTA NUM DESLOCAMENTO ≠ 0. MEDE TAMBÉM REJEITO/PERÍODO.
import pickle, numpy as np, pandas as pd, cv2
cv2.setNumThreads(4)
R = '/home/grva-mint/Projects/Falhas'
GAP, SIDE, HALF, LAGS = 4, 12, 20, 14     # FOLGA ATÉ A LINHA, TRAÇOS DE CADA LADO, MEIA-JANELA VERTICAL, DESLOCAMENTO MÁXIMO

def gauss(img, s):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT)

def slope(sec):
    sm = gauss(sec, 1.0)
    gz, gx = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    Jzz, Jxx, Jzx = gauss(gz * gz, 6.0), gauss(gx * gx, 6.0), gauss(gz * gx, 6.0)
    return np.clip(-np.tan(0.5 * np.arctan2(2 * Jzx, Jzz - Jxx)), -2, 2)

def period(sec, z, x):
    w = sec[max(0, z - 64):z + 64, max(0, x - 32):x + 32]
    w = w - w.mean(0)
    p = (np.abs(np.fft.rfft(w, axis=0)) ** 2).mean(1)[1:]
    f = np.fft.rfftfreq(w.shape[0])[1:]
    return float(p.sum() / ((f * p).sum() + 1e-12))

# TRAÇO COMPOSTO DE UM LADO: CADA TRAÇO É LEVADO ATÉ A FALHA PELO MERGULHO LOCAL (dz = slope·dx) E A MÉDIA SAI NA JANELA
def composite(sec, dip, z, xf, sign):
    zz = np.arange(z - HALF - LAGS, z + HALF + LAGS + 1)
    acc = []
    for k in range(GAP, GAP + SIDE):
        x = int(round(xf + sign * k))
        if x < 0 or x >= sec.shape[1]:
            continue
        shift = dip[np.clip(zz, 0, sec.shape[0] - 1), x] * (sign * k)     # PROFUNDIDADE ONDE A CAMADA QUE CHEGA EM z ESTÁ NESTE TRAÇO
        acc.append(np.interp(zz + shift, np.arange(sec.shape[0]), sec[:, x]))
    return np.mean(acc, 0) if acc else None

def evidence(sec, dip, s, step=10):
    p0, p1 = np.array(s['p0'], float), np.array(s['p1'], float)
    n = max(int(np.hypot(*(p1 - p0)) / step), 1)
    rows = []
    for t in np.linspace(0.05, 0.95, n):
        x, z = p0 + t * (p1 - p0)
        z = int(round(z))
        if z - HALF - LAGS < 0 or z + HALF + LAGS >= sec.shape[0]:
            continue
        a, b = composite(sec, dip, z, x, -1), composite(sec, dip, z, x, +1)
        if a is None or b is None or a.std() < 1e-4 or b.std() < 1e-4:
            continue
        core = a[LAGS:LAGS + 2 * HALF + 1]
        cc = []
        for lag in range(-LAGS, LAGS + 1):
            seg = b[LAGS + lag:LAGS + lag + 2 * HALF + 1]
            cc.append(np.corrcoef(core, seg)[0, 1])
        cc = np.array(cc)
        best = int(np.argmax(cc))
        rows.append({'rho0': cc[LAGS], 'rho': cc[best], 'lag': best - LAGS, 'per': period(sec, z, int(x))})
    return pd.DataFrame(rows)

def summary(df):
    if not len(df):
        return {'pontos': 0}
    hit = (df.rho > 0.6) & (df.lag.abs() >= 2) & (df.rho - df.rho0 > 0.15)
    return {'pontos': len(df), 'rho0': df.rho0.median(), 'rho': df.rho.median(), 'lag_abs': df.lag.abs().median(), 'frac_rejeito': hit.mean(), 'rejeito_periodo': float((df.lag.abs() / df.per)[hit].median()) if hit.any() else np.nan, 'periodo': df.per.median()}


if __name__ == '__main__':
    store = pickle.load(open('profile/art5_sticks_dev.pkl', 'rb'))
    table = pd.read_pickle('profile/art5_sticks_dev_table.pkl')
    audit = pd.read_csv('profile/audit/audit_categorias.csv', index_col=0)
    loose = table[table['cov'] < 0.2].sample(frac=1, random_state=0).reset_index().iloc[:48]
    secs  = {pid: np.load(f'{R}/Blocks/Marlim/files/cache/real_center_{pid}.npy').astype(np.float32) - 0.50263 for pid in ('1200', '1300')}
    dips  = {pid: slope(s) for pid, s in secs.items()}
    out   = []
    for pid in ('1200', '1300'):
        for g in store[pid]['gt']:
            zc = (g['p0'][1] + g['p1'][1]) / 2
            if 450 <= zc < 1200:
                out.append({'grupo': 'especialista', 'patch': pid, 'zc': zc, 'len': g['len'], **summary(evidence(secs[pid], dips[pid], g))})
    first = {pid: table.index[table.patch == pid][0] for pid in ('1200', '1300')}
    for i, row in loose.iterrows():
        s = store[row.patch]['mine'][row['index'] - first[row.patch]]
        out.append({'grupo': 'FP_' + audit.cat.iloc[i], 'patch': row.patch, 'zc': row.zc, 'len': s['len'], **summary(evidence(secs[row.patch], dips[row.patch], s))})
    df = pd.DataFrame(out)
    df.to_pickle('profile/offset_evidence.pkl')
    pd.set_option('display.width', 200)
    print(df.groupby('grupo').agg(n=('len', 'size'), frac_rejeito=('frac_rejeito', 'median'), frac_rejeito_media=('frac_rejeito', 'mean'), rho0=('rho0', 'median'), rho=('rho', 'median'), lag_abs=('lag_abs', 'median'), rejeito_periodo=('rejeito_periodo', 'median'), periodo=('periodo', 'median')).round(3).to_string())
