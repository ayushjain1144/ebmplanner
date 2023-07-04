import torchvision.transforms as T
import torch

transform = T.Compose([
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, img_h, img_w):
    out_bbox[:, 2:] = torch.clamp(out_bbox[:, 2:], min=0.01, max=1.0)
    b = box_cxcywh_to_xyxy(out_bbox)
    b = torch.clamp(b, min=0.0, max=1.0)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b