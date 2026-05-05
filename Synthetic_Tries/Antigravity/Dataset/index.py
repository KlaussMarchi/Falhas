import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages

class VolumeLoader:
    """Handles loading and formatting of numpy volume files."""
    @staticmethod
    def load(path):
        if not os.path.exists(path):
            return None
        vol = np.load(path)
        # Squeeze to handle any extra dimensions (ensure they are 3D)
        if vol.ndim > 3:
            vol = np.squeeze(vol)
        return vol

class Slicer:
    """Handles extracting 2D slices from a 3D volume."""
    @staticmethod
    def get_mid_slices(vol):
        mid_x = vol.shape[0] // 2
        mid_y = vol.shape[1] // 2
        mid_z = vol.shape[2] // 2

        slices = [
            vol[mid_x, :, :],  # YZ plane (Slice X)
            vol[:, mid_y, :],  # XZ plane (Slice Y)
            vol[:, :, mid_z]   # XY plane (Slice Z)
        ]
        return slices, (mid_x, mid_y, mid_z)

class PDFReportGenerator:
    """Orchestrates the loading, plotting, and PDF generation."""
    def __init__(self, ds0_img_dir, ds0_mask_dir, ds1_img_dir, ds1_mask_dir, output_pdf):
        self.ds0_img_dir = ds0_img_dir
        self.ds0_mask_dir = ds0_mask_dir
        self.ds1_img_dir = ds1_img_dir
        self.ds1_mask_dir = ds1_mask_dir
        self.output_pdf = output_pdf

    def _get_file_paths(self, idx):
        ds0_img_path = os.path.join(self.ds0_img_dir, f'{idx}.npy')
        ds0_mask_path = os.path.join(self.ds0_mask_dir, f'{idx}.npy')
        
        # ds1 files might be img_{idx:04d}.npy or {idx}.npy
        ds1_img_path = os.path.join(self.ds1_img_dir, f'img_{idx:04d}.npy')
        if not os.path.exists(ds1_img_path):
            ds1_img_path = os.path.join(self.ds1_img_dir, f'{idx}.npy')
            
        ds1_mask_path = os.path.join(self.ds1_mask_dir, f'img_{idx:04d}.npy')
        if not os.path.exists(ds1_mask_path):
            ds1_mask_path = os.path.join(self.ds1_mask_dir, f'{idx}.npy')
            
        return ds0_img_path, ds0_mask_path, ds1_img_path, ds1_mask_path

    def _plot_row(self, axes, row_idx, vol, label, is_mask, overlay_mask_vol=None):
        slices, (mid_x, mid_y, mid_z) = Slicer.get_mid_slices(vol)
        
        # Ajuste das fatias originais
        slices[0] = np.array(slices[0])
        arr_y = np.array(slices[1])
        arr_z = np.array(slices[2])
        slices[1] = np.rot90(arr_z, -1)
        slices[2] = arr_y
        
        # Se houver uma máscara para sobrepor (Paste = True)
        if overlay_mask_vol is not None:
            mask_slices, _ = Slicer.get_mid_slices(overlay_mask_vol)
            mask_slices[0] = np.array(mask_slices[0])
            m_arr_y = np.array(mask_slices[1])
            m_arr_z = np.array(mask_slices[2])
            mask_slices[1] = np.rot90(m_arr_z, -1)
            mask_slices[2] = m_arr_y
    
        if is_mask:
            # Fundo preto com falhas brancas
            cmap_config = ListedColormap(['black', 'white', 'white', 'white'])
            vmin, vmax = (0, 3)
        else:
            cmap_config = 'gray'
            vmin, vmax = (None, None)
        
        titles = [f'{label}\nSlice X={mid_x} (inline)', f'{label}\nSlice Y={mid_y} (crossline)', f'{label}\nSlice Z={mid_z} (depth)']
        
        for col_idx, ax in enumerate(axes[row_idx]):
            # Plota a imagem base (sísmica ou máscara)
            ax.imshow(slices[col_idx], cmap=cmap_config, vmin=vmin, vmax=vmax)
            
            # Lógica do Overlay (Paste)
            if overlay_mask_vol is not None:
                # Transforma a máscara em binária (1 para falha, 0 para fundo)
                m_slice = mask_slices[col_idx]
                m_slice_bin = (m_slice > 0).astype(int)
                
                # Esconde os zeros (para ficarem transparentes) e plota só as falhas
                m_slice_masked = np.ma.masked_where(m_slice_bin == 0, m_slice_bin)
                
                # Mapa de cores contendo apenas vermelho
                red_cmap = ListedColormap(['red'])
                # alpha controla a transparência do vermelho (1.0 = sólido)
                ax.imshow(m_slice_masked, cmap=red_cmap, alpha=0.8, interpolation='none')
            
            ax.set_title(titles[col_idx], fontsize=12)
            ax.axis('off')

    def generate(self, num_samples=25):
        with PdfPages(self.output_pdf) as pdf:
            for i in range(num_samples):
                paths = self._get_file_paths(i)
                
                if not all(os.path.exists(p) for p in paths):
                    print(f"Skipping index {i} due to missing file(s).")
                    continue
                    
                ds0_img, ds0_mask, ds1_img, ds1_mask = [VolumeLoader.load(p) for p in paths]

                data_list = [
                    (ds0_img,  'Wu Synthetic Seismic (Dataset 0)', False, None),
                    (ds0_mask, 'Wu Synthetic Fault (Dataset 0)', True, None),
                    (ds1_img,  'GRvA Synthetic Seismic (Dataset 1)', False, None),
                    (ds1_mask, 'GRvA Synthetic Fault (Dataset 1)', True, None),
                ]

                data_list = [
                    (ds0_img,  'Wu Synthetic Seismic (Dataset 0)', False, None),
                    (ds0_img, 'Wu Synthetic Fault (Dataset 0)', False, ds0_mask),
                    (ds1_img,  'GRvA Synthetic Seismic (Dataset 1)', False, None),
                    (ds1_img, 'GRvA Synthetic Fault (Dataset 1)', False, ds1_mask),
                ]

                # Ajusta dinamicamente a altura e a quantidade de linhas
                num_rows = len(data_list)
                fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
                #fig.suptitle(f'Comparison for Sample {i}', fontsize=20)
                
                for row_idx, (vol, label, is_mask, overlay_mask_vol) in enumerate(data_list):
                    self._plot_row(axes, row_idx, vol, label, is_mask, overlay_mask_vol)
                
                plt.tight_layout()
                # Ajusta o título com base na quantidade de gráficos para não sobrepor
                plt.subplots_adjust(top=1 - (0.15 / num_rows))
                
                pdf.savefig(fig)
                plt.close(fig)
                print(f"Processed sample {i}")

        print(f"Successfully created {self.output_pdf}")


if __name__ == "__main__":
    generator = PDFReportGenerator(
        ds0_img_dir='dataset0/images',
        ds0_mask_dir='dataset0/masks',
        ds1_img_dir='dataset1/original/images',
        ds1_mask_dir='dataset1/original/masks',
        output_pdf='output.pdf'
    )
    generator.generate(num_samples=15)