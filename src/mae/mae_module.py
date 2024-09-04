import torch.nn as nn
from transformers import ViTImageProcessor, ViTMAEConfig
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from objmask.mae.trainer import CustomViTMAEForPreTraining
from PIL import Image
import torch
import torchvision.transforms as T

class CustomMae(nn.Module):
    def __init__(self, mae_path, device):
        super(CustomMae, self).__init__()
        self.device = device
        self.feature_extractor = ViTImageProcessor.from_pretrained(mae_path)
        self.imagenet_mean = np.array(self.feature_extractor.image_mean)
        self.imagenet_std = np.array(self.feature_extractor.image_std)
        self.config = ViTMAEConfig()
        self.config.update(
            {
                "mask_ratio": -1,
                "_name_or_path": mae_path,
            }
        )
        self.transform = Compose([Resize((224, 224)), ToTensor()])
        self.transform_mask = Compose([lambda image: image.convert("1"), Resize((224, 224)), ToTensor()])
        self.model = CustomViTMAEForPreTraining.from_pretrained(mae_path, config=self.config).to(self.device)
        # frozen the model
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, image, mask):
        self.image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        mask_tensor = self.transform_mask(mask).to(self.device)
        outputs = self.model(self.image, mask_tensor)
        return outputs
    
if __name__ == "__main__":
    mae = CustomMae("checkpoint/mae/checkpoint-2690-new")
    img = Image.open("data/MISATO/input/0151.png")
    mask = Image.open("data/MISATO/input/0151_mask.png")
    outputs = mae(img, mask)
    print(outputs.logits.shape)
    y = mae.model.unpatchify(outputs.logits)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    mask = outputs.mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, mae.model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
    mask = mae.model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', mae.image)
    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask
    img_paste_pil = torch.clip((im_paste[0] * mae.imagenet_std + mae.imagenet_mean) * 255, 0, 255).to(torch.uint8)
    img_paste_pil = img_paste_pil.permute(2, 0, 1)
    T.ToPILImage()(img_paste_pil).save("Test_mae_output.png")
        