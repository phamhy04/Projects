
import torch.nn as nn
import torch
import math
from torchvision.ops import nms
from retinanet.utils import BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # Upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)    #   Size same, depth decrease: 2048 -> 256
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')     #   Double size of feature-map

        # Add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)     #   Size same, depth decrease: 1024 -> 256
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # Add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)     #   Size same, depth decrease: 512 -> 256

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)      #   Half size, decrease depth: 2048 -> 256

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)      #   Half size, depth the same

    def forward(self, inputs):
        C3, C4, C5 = inputs #   28x28x512, 14x14x1024, 7x7x2048 respectively
        P5_x = self.P5_1(C5)    #   7x7x2048 -> ======= 7x7x256 =======
        P5_upsampled_x = self.P5_upsampled(P5_x)    #   7x7x256 -> 14x14x256

        P4_x = self.P4_1(C4)    #   14x14x1024 -> 14x14x256
        P4_x = P5_upsampled_x + P4_x    #   14x14x256 + 14x14x256 = ======= 14x14x256 =======
        P4_upsampled_x = self.P4_upsampled(P4_x)    #   14x14x256 -> 28x28x256

        P3_x = self.P3_1(C3)    #   28x28x512 -> 28x28x256
        P3_x = P3_x + P4_upsampled_x    #   28x28x256 + 28x28x256 = 28x28x256

        P6_x = self.P6(C5)  #   7x7x2048 -> 3.5x3.5x256

        P7_x = self.P7_1(P6_x)  #   ====== ReLU ======
        P7_x = self.P7_2(P7_x)  #   ====== 2x2x256 ======

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)
        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class AttributeModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_attr=40, prior=0.01, feature_size=256):
        super(AttributeModel, self).__init__()

        self.num_attr = num_attr
        self.num_anchors = num_anchors

        self.conv = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_attr, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.act(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = num_attr * num_anchor
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_attr)

        return out2.contiguous().view(x.shape[0], -1, self.num_attr)


class BackBone(nn.Module):
    def __init__(self, num_attr):

        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        fpn_sizes = [self.conv2.out_channels,
                     self.conv3.out_channels,
                     self.conv4.out_channels]

        #   Build (FPN) Feature Pyramid Network
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        #   Bounding Box Regression
        self.regressionModel = RegressionModel(256)
        #   Attributes
        self.attributeModel = AttributeModel(256, num_attr=num_attr)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.attributeModel.output.weight.data.fill_(0)
        self.attributeModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        x1 = self.conv1(img_batch)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x1 = self.maxpool(x1)

        x2 = self.conv2(x1)
        # x2 = torch.cat((x2, torch.zeros([x2.shape[0], x2.shape[1], x2.shape[2], 1]).to('cuda')), dim=3)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)


        #   Feature Pyramid Network (FPN)
        #   x2, x3, x4 are the outputs with depth of 512, 1024, 2048 respectively with ResNet_50
        features = self.fpn([x2, x3, x4])
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        attribute = torch.cat([self.attributeModel(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(attribute, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(attribute.shape[2]):
                scores = torch.squeeze(attribute[:, :, i])
                scores = torch.squeeze(attribute[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
