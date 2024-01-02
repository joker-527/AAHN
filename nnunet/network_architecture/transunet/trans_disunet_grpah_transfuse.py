import torch.nn as nn
from einops import rearrange
import torch

from nnunet.network_architecture.Attention import DisTransBlock, SE
from nnunet.network_architecture.transunet.bottleneck_layer import Bottleneck
from nnunet.network_architecture.transunet.decoder import Up, SignleConv
from nnunet.network_architecture.vit.vit import ViT




class Resconv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, out_ch):
        super(Resconv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        init = x
        x = self.conv(x)
        out = x + init
        return out


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class Resconv(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch, num_layers):
        super(Resconv, self).__init__()
        self.num_layers = num_layers
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = conv_block(in_ch, out_ch)
        self.block_list = nn.ModuleList([Resconv_block(
            out_ch=out_ch
        ) for i in range(num_layers)])

    def forward(self, x):
        init = self.conv1(x)
        x = self.conv2(x)
        for block in self.block_list:
            x = block(x)
        out = x + init
        return out



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.drop = nn.Dropout2d(0.3)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.drop(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.drop(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop(x)
        x = self.relu3(x)
        return x

class Decoder_skip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_skip, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.DoubleConv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, down_input, skip_input):
        x = self.transconv(down_input)
        x1 = torch.cat([x, skip_input], dim=1)
        x1 = self.DoubleConv(x1)

        return x1


class GraphLayer(nn.Module):
    def __init__(self, num_state, num_node, num_class):
        super().__init__()
        self.vis_gcn = GCN(num_state, num_node)
        self.word_gcn = GCN(num_state, num_class)
        self.transfer = GraphTransfer(num_state)
        self.gamma_vis = torch.zeros([num_node])
        self.gamma_word = torch.zeros([num_class])
        self.gamma_vis = nn.Parameter(
            self.gamma_vis,
            requires_grad=True)
        self.gamma_word = nn.Parameter(
            self.gamma_vis,
            requires_grad=True)

    def forward(self, inp, vis_node):
        inp_1 = inp
        batch_size, in_channels, _, _ = inp_1.shape
        inp = self.word_gcn(inp)
        new_V = self.vis_gcn(vis_node)

        class_node, vis_node = self.transfer(inp, new_V)
        class_node = self.gamma_word * inp + class_node
        class_node = class_node.view(batch_size, in_channels, *inp_1.size()[2:])  # [1, 3, 4096] --- > [1, 3, 64, 64]
        class_node = class_node + inp_1
        return class_node


class GCN(nn.Module):
    def __init__(self, num_state=128, num_node=64, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(
            num_node,
            num_node,
            kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            num_state,
            num_state,
            kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        # (N, C, THW)
        x = x.view(batch_size, x.size()[1], -1)
        x = x.permute(0, 2, 1)
        h = self.conv1(x)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        h = h.permute(0, 2, 1)
        # print(h.shape)#torch.Size([1, 3, 4096])
        return h


class GraphTransfer(nn.Module):
    """Transfer vis graph to class node, transfer class node to vis feature"""

    def __init__(self, in_dim):
        super().__init__()
        self.channle_in = in_dim
        self.query_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv_vis = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv_word = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.softmax_vis = nn.Softmax(dim=-1)
        self.softmax_word = nn.Softmax(dim=-2)

        self.lastconv_vis = nn.Conv1d(
            in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1)
        self.lastconv_word = nn.Conv1d(
            in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1)

    def forward(self, word, vis_node):
        m_batchsize, C, Nc = word.shape
        m_batchsize, C, Nn = vis_node.shape

        word = word.permute(0, 2, 1)
        vis_node = vis_node.permute(0, 2, 1)

        proj_query = self.query_conv(word)
        proj_query = proj_query.view(m_batchsize, C, -1)
        proj_query = proj_query.permute((0, 2, 1))

        proj_key = self.key_conv(vis_node).view(m_batchsize, C, -1)

        energy = torch.matmul(proj_query, proj_key)
        attention_vis = self.softmax_vis(energy).permute((0, 2, 1))
        attention_word = self.softmax_word(energy)

        proj_value_vis = self.value_conv_vis(vis_node).view(m_batchsize, C, -1)
        proj_value_word = self.value_conv_word(word).view(m_batchsize, C, -1)

        class_out = torch.matmul(proj_value_vis, attention_vis).contiguous()
        node_out = torch.matmul(proj_value_word, attention_word).contiguous()

        class_out = self.lastconv_vis(class_out.permute((0, 2, 1))).permute((0, 2, 1))
        node_out = self.lastconv_word(node_out.permute((0, 2, 1))).permute((0, 2, 1))
        return class_out, node_out




class TransUnet_dis_graph_transfuse(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_blocks=12,
                 vit_heads=12,
                 vit_dim_linear_mhsa_block=3072,
                 patch_size=8,
                 vit_transformer_dim=768,
                 vit_transformer=None,
                 ):
        super().__init__()
        self.inplanes = 128
        self.patch_size = patch_size
        self.vit_transformer_dim = vit_transformer_dim
        self.in_channels = in_channels

        vit_channels1 = self.inplanes * 2
        vit_channels2 = self.inplanes * 4
        vit_channels3 = self.inplanes * 8

        # Not clear how they used resnet arch. since the first input after conv
        # must be 128 channels and half spat dims.

        in_conv1 = nn.Conv2d(classes, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.trans_init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))

        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.vit_conv1 = SignleConv(in_ch=vit_channels1, out_ch=self.inplanes * 2)

        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.vit_conv2 = SignleConv(in_ch=vit_channels2, out_ch=self.inplanes * 4)


        self.conv3 = Bottleneck(self.inplanes * 4, self.inplanes * 8, stride=2)
        self.vit_conv3 = SignleConv(in_ch=vit_channels3, out_ch=self.inplanes * 8)

        self.down = nn.MaxPool2d(kernel_size=2)

        self.img_dim_vit1 = img_dim // 4
        self.img_dim_vit2 = img_dim // 8
        self.img_dim_vit3 = img_dim // 16

        assert (self.img_dim_vit3 % patch_size == 0), "Vit patch_dim not divisible"

        self.vit1 = ViT(img_dim=self.img_dim_vit1,
                       in_channels=vit_channels1,  # input features' channels (encoder)
                       patch_dim=patch_size,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        self.vit2 = ViT(img_dim=self.img_dim_vit2,
                       in_channels=vit_channels2,  # input features' channels (encoder)
                       patch_dim=patch_size,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer

        self.vit3 = ViT(img_dim=self.img_dim_vit3,
                       in_channels=vit_channels3,  # input features' channels (encoder)
                       patch_dim=patch_size,
                       # transformer inside dimension that input features will be projected
                       # out will be [batch, dim_out_vit_tokens, dim ]
                       dim=vit_transformer_dim,
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False) if vit_transformer is None else vit_transformer


        # to project patches back - undoes vit's patchification
        token_dim1 = vit_channels1 * (patch_size ** 2)
        token_dim2 = vit_channels2 * (patch_size ** 2)
        token_dim3 = vit_channels3 * (patch_size ** 2)


        self.project_patches_back1 = nn.Linear(vit_transformer_dim, token_dim1)
        self.project_patches_back2 = nn.Linear(vit_transformer_dim, token_dim2)
        self.project_patches_back3 = nn.Linear(vit_transformer_dim, token_dim3)


        self.conv_init_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True))

        self.layer1_Conv = Resconv(self.inplanes, self.inplanes, 3)
        self.layer2_Conv = Resconv(self.inplanes * 2, self.inplanes * 2, 4)
        self.layer3_Conv = Resconv(self.inplanes * 4, self.inplanes * 4, 6)
        self.layer4_Conv = Resconv(self.inplanes * 8, self.inplanes * 8, 2)

        self.mutual_trans1 = DisTransBlock(self.inplanes, dimension=2)
        self.mutual_trans2 = DisTransBlock(self.inplanes * 2, dimension=2)
        self.mutual_trans3 = DisTransBlock(self.inplanes * 4, dimension=2)
        self.mutual_trans4 = DisTransBlock(self.inplanes * 8, dimension=2)

        self.trans_se1 = SE(self.inplanes * 2, 8)
        self.trans_se2 = SE(self.inplanes * 4, 8)
        self.trans_se3 = SE(self.inplanes * 8, 8)
        self.trans_se4 = SE(self.inplanes * 16, 8)

        self.delchl1 = Resconv(self.inplanes * 2, self.inplanes, 3)
        self.delchl2 = nn.Conv2d(self.inplanes * 4, self.inplanes * 2, 1)
        self.delchl3 = nn.Conv2d(self.inplanes * 8, self.inplanes * 4, 1)
        self.delchl4 = nn.Conv2d(self.inplanes * 16, self.inplanes * 8, 1)

        self.Convtdown1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.inplanes * 2),
            nn.LeakyReLU()
        )
        self.Convtdown2 = nn.Sequential(
            nn.Conv2d(self.inplanes * 2, self.inplanes * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.inplanes * 4),
            nn.LeakyReLU()
        )
        self.Convtdown3 = nn.Sequential(
            nn.Conv2d(self.inplanes * 4, self.inplanes * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.inplanes * 8),
            nn.LeakyReLU()
        )


        # upsampling path
        self.dec1 = Decoder_skip(1024, 512)
        self.dec2 = Decoder_skip(512, 256)
        self.dec3 = Decoder_skip(256, 128)
        self.dec4 = DecoderBlock(128, 64)

        self.finaldeconv1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.finaldeconv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.GraphTrans = GraphLayer(64, 64, 64)

        self.conv1x1 = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=1)

    def forward(self, x):
        x_t1 = x[:, 0, :, :]
        x_t2 = x[:, 1, :, :]
        x_ct = x[:, 2, :, :]  # t2
        x_pet = x[:, 3, :, :]  # flair
        x_ct1 = torch.repeat_interleave(x_t2.unsqueeze(dim=1), repeats=self.in_channels, dim=1)

        trans_x1 = self.trans_init_conv(x)#[2, 128, 64, 64]
        trans_x2 = self.conv1(trans_x1)#[2, 256, 32, 32]
        trans_vit1 = self.vit1(trans_x2)
        trans_vit1 = self.project_patches_back1(trans_vit1)
        trans_vit1 = rearrange(trans_vit1, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                      x=self.img_dim_vit1 // self.patch_size, y=self.img_dim_vit1 // self.patch_size,
                      patch_x=self.patch_size, patch_y=self.patch_size)
        trans_vit1 = self.vit_conv1(trans_vit1)#[2, 256, 32, 32]

        trans_x3 = self.conv2(trans_vit1)#[2, 512, 16, 16]
        trans_vit2 = self.vit2(trans_x3)
        trans_vit2 = self.project_patches_back2(trans_vit2)
        trans_vit2 = rearrange(trans_vit2, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                               x=self.img_dim_vit2 // self.patch_size, y=self.img_dim_vit2 // self.patch_size,
                               patch_x=self.patch_size, patch_y=self.patch_size)
        trans_vit2 = self.vit_conv2(trans_vit2)  # [2, 512, 16, 16]


        trans_x4 = self.conv3(trans_vit2) # [2, 1024, 8, 8])
        trans_vit3 = self.vit3(trans_x4)
        trans_vit3 = self.project_patches_back3(trans_vit3)
        trans_vit3 = rearrange(trans_vit3, 'b (x y) (patch_x patch_y c) -> b c (patch_x x) (patch_y y)',
                               x=self.img_dim_vit3 // self.patch_size, y=self.img_dim_vit3 // self.patch_size,
                               patch_x=self.patch_size, patch_y=self.patch_size)
        trans_vit3 = self.vit_conv3(trans_vit3)  # [2, 1024, 8, 8])

        conv_x1 = self.conv_init_conv(x_ct1)#[2, 128, 64, 64]

        fuse1 = self.trans_se1(torch.cat([conv_x1,trans_x1],dim=1))
        fuse1 = self.delchl1(fuse1)
        #fuse1 = self.mutual_trans1(conv_x1,fuse1)
        conv_x2 = self.layer1_Conv(fuse1 + conv_x1)

        conv_x3 = self.Convtdown1(conv_x2)#[2, 256, 32, 32]

        fuse2 = self.trans_se2(torch.cat([conv_x3, trans_vit1], dim=1))
        fuse2 = self.delchl2(fuse2)
        fuse2 = self.mutual_trans2(conv_x3, fuse2)
        conv_x3 = self.layer2_Conv(fuse2)

        conv_x4 = self.Convtdown2(conv_x3)  # [2, 512, 16, 16]

        fuse3 = self.trans_se3(torch.cat([conv_x4, trans_vit2], dim=1))
        fuse3 = self.delchl3(fuse3)
        fuse3 = self.mutual_trans3(conv_x4, fuse3)
        conv_x4 = self.layer3_Conv(fuse3)

        conv_x5 = self.Convtdown3(conv_x4)  # [2, 1024, 8, 8]

        fuse4 = self.trans_se4(torch.cat([conv_x5, trans_vit3], dim=1))
        fuse4 = self.delchl4(fuse4)
        fuse4 = self.mutual_trans4(conv_x5, fuse4)
        conv_x5 = self.layer4_Conv(fuse4)

        graph_fuse = self.GraphTrans(conv_x5, trans_vit3)

        y = self.dec1(graph_fuse, conv_x4)#[2, 512, 16, 16]
        y = self.dec2(y, conv_x3)#[2, 256, 32, 32]
        y = self.dec3(y, conv_x2)
        y = self.dec4(y)

        out = self.finaldeconv1(y)
        out = self.finaldeconv2(out)
        return self.conv1x1(out)


if __name__ == "__main__":
    import torch
    a = torch.rand(2, 4, 128, 128)
    model = TransUnet_dis_graph_transfuse(in_channels=3, img_dim=128, vit_blocks=8,
    vit_dim_linear_mhsa_block=512, classes=4)
    y = model(a) # [2, 5, 128, 128]
    print(y.shape)