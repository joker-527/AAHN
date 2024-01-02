import torch.nn as nn
from nnunet.network_architecture.transunet.trans_disunet_grpah_transfuse import TransUnet_dis_graph_transfuse


class custom_net(SegmentationNetwork):

    def __init__(self):
        super(custom_net, self).__init__()
        self.params = {'content': None}
        self.conv_op = nn.Conv2d
        self.do_ds = False
        #self.num_classes = num_classes

        ######## self.model 设置自定义网络 by Sleeep ########
        self.model = TransUnet_dis_graph_transfuse(
            in_channels=3, 
            img_dim=128, 
            vit_blocks=8,
            vit_dim_linear_mhsa_block=512, 
            classes=4)

    def forward(self, x):
        if self.do_ds:
            return [self.model(x), ]
        else:
            return self.model(x)


def create_model():
    return custom_net()
