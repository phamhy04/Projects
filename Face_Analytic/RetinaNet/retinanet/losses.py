import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    #   Discard all of value Ypred_bbox & Ytr_bbox that don't overlap together
    iw = torch.clamp(iw, min=0)     #   Shape: (Num_anchor_bbox, num_ground_truth_bbox)
    ih = torch.clamp(ih, min=0)     #   Shape: (Num_anchor_bbox, num_ground_truth_bbox)

    union = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    union = torch.clamp(union, min=1e-8)
    intersection = iw * ih

    IoU = intersection / union

    return IoU

class FocalLoss(nn.Module):

    def forward(self, attributes, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = attributes.shape[0]
        attribute_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5*anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5*anchor_heights

        for j in range(batch_size):

            attribute = attributes[j, :, :]
            regression = regressions[j]
            bbox_annotation = annotations[j].unsqueeze(dim=0)

            attribute = torch.clamp(attribute, 1e-4, 1.0 - 1e-4)
            IoU = calc_iou(anchor, bbox_annotation[:, :4]) # num_anchor x num_annotation

            # Compute the loss for classification
            targets = torch.ones(attribute.shape) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU.squeeze(), 0.4), :] = 0  #   torch.lt: Computes IoU_max < 0.4 element-wise
            #   Those anchors that have IoU_max with gt_bbox >= 0.5
            positive_indices = torch.ge(IoU.squeeze(), 0.5)  #   torch.lt: Computes IoU_max >= 0.5 element-wise
            num_positive_anchors = positive_indices.sum()

            #   Pick out bbox_annotations that have largest IoU with specific anchors
            assigned_annotations = bbox_annotation.repeat(IoU.shape[0], 1)

            #   Create labels for attribute
            targets[positive_indices] = bbox_annotation[0, 4:]

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha
#
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - attribute, attribute)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(attribute) + (1.0 - targets) * torch.log(1.0 - attribute))

            attribute_loss = focal_weight * bce

            if torch.cuda.is_available():
                attribute_loss = torch.where(torch.ne(targets, -1.0), attribute_loss, torch.zeros(attribute_loss.shape).cuda())
            else:
                attribute_loss = torch.where(torch.ne(targets, -1.0), attribute_loss, torch.zeros(attribute_loss.shape))
            attribute_losses.append(attribute_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # Compute the loss for bbox regression
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(attribute_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


