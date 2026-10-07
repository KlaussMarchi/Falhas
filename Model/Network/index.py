import torch.nn.functional as F
import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import BinaryJaccardIndex
from monai.networks.nets import SegResNet

from .types.UNet3D import UNet3D
from .types.Unet3D_V2 import Unet3D_V2
from .types.ResACEUnet import ResACEUnet
from .types.ResACEUnet_Wu import ResACEUnetWu
from .types.ResACEUnet_Zu import ResACEUnetZu
from .types.FaultSegNet import FaultSegNet
from .types.NRUNet import NRUNet
from .types.MACNN  import MACNN
from .types.FaultEdgeFormer import FaultEdgeFormer


class ModelNetwork:
    selected = None

    def __init__(self, network, img_size, classes=1, channels=1, lr=1e-4, dropout=0.1, num_filters=16):
        self.network = network
        self.img_size = img_size
        self.classes  = classes
        self.multiclass  = (self.classes > 1)
        self.channels    = channels
        self.dropout     = dropout
        self.num_filters = num_filters
        self.lr = lr
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = self.get().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        if self.multiclass:
            self.iou = MulticlassJaccardIndex(num_classes=self.classes, average='macro').to(self.device)
        else:
            self.iou = BinaryJaccardIndex(threshold=0.5).to(self.device)
    
    def get(self):
        classes = self.classes

        # UNET PADRÃO TESTADA NO PROJETO DOS DENTISTAS
        if self.network == 'unet_3d':
            return UNet3D(img_channels=self.channels, num_filters=self.num_filters, dropout=self.dropout, classes=classes)

        # UNET 3D MODIFICADA GRVA (LUCAS E CELIA)
        if self.network == 'dbrnet':
            return Unet3D_V2(img_channels=self.channels, classes=classes, num_filters=self.num_filters, dropout=self.dropout)

        # TESTADA PARA PROJETOS DE SEGMENTAÇÃO NA AREA METIRCA
        if self.network == 'segresnet':
            return SegResNet(spatial_dims=3, in_channels=self.channels, out_channels=self.classes, init_filters=self.num_filters, dropout_prob=self.dropout)

        # RESACUNET MODIFICADA COM BOTTLENECK E PROFUNDIDADE (ZU ET AL. 2024, ResACEUnet...pdf)
        if self.network == 'resaceunet_grva':
            return ResACEUnet(in_channels=self.channels, num_classes=classes, base_filters=self.num_filters, dropout_rate=self.dropout, input_shape=self.img_size)
        
        # RESACEUNET ORIGINAL DO REPOSITÓRIO DOS AUTORES (ZU ET AL. 2024, github.com/39c5bb-miku/ResACEUnet)
        if self.network == 'resaceunet_wu':
            return ResACEUnetWu(in_channels=self.channels, out_channels=classes, img_size=self.img_size[0], feature_size=self.num_filters, hidden_size=self.num_filters * 16, dims=[self.num_filters * m for m in (2, 4, 8, 16)], drop_rate=self.dropout)

        # RESACEUNET DO ARTIGO, ENCODER DE 3 ESTÁGIOS (ZU ET AL. 2024, PRIMEIRO COMMIT DE 05/12/2024 EM github.com/39c5bb-miku/ResACEUnet)
        if self.network == 'resaceunet_zu':
            return ResACEUnetZu(in_channels=self.channels, out_channels=classes, img_size=self.img_size, feature_size=self.num_filters, hidden_size=self.num_filters * 32, dims=[self.num_filters * m for m in (2, 4, 32)], dropout_rate=self.dropout)

        # MACNN MULTIESCALA COM ATENÇÃO (GAO ET AL. 2022, 3 - Automatic fault detection...pdf)
        if self.network == 'macnn':
            return MACNN(in_channels=self.channels, num_classes=classes, base_filters=self.num_filters, dropout_rate=self.dropout, input_shape=self.img_size)

        # FAULT-SEG-NET COM FUSÃO MULTIESCALA (LI ET AL. 2023, 1 - Fault-Seg-Net A method for...pdf)
        if self.network == 'fault_seg_net':
            return FaultSegNet(in_channels=self.channels, num_classes=classes, base_filters=self.num_filters, dropout_rate=self.dropout, input_shape=self.img_size)

        # U-NET RESIDUAL ANINHADA (GAO ET AL. 2022, Fault_Detection_on_..._Nested_Residual_U-Net.pdf)
        if self.network == 'nru_net':
            return NRUNet(in_channels=self.channels, num_classes=classes, base_filters=self.num_filters, dropout_rate=self.dropout, input_shape=self.img_size)

        # TRANSFORMER COM SOBEL TREINÁVEL NAS BORDAS (DI ET AL. 2026, 4 - FaultEdgeFormer_...pdf)
        if self.network == 'fault_edge_former':
            return FaultEdgeFormer(in_channels=self.channels, num_classes=classes, base_filters=self.num_filters, dropout_rate=self.dropout, input_shape=self.img_size)

        return None