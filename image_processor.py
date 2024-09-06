from torchvision import transforms
import torch
import numpy as np
import albumentations

# convert pil image to tensor
def pil_to_tensor(pil_image):
    compose = transforms.Compose([transforms.ToTensor()])
    return transforms.ToTensor()(pil_image)
    # compose = transforms.Compose(
    #     [
    #         if pil_image.mode != "L":
    #             pil_image = pil_image.convert("L")
    #         transforms.ToTensor()
    #     ]
    # )
    # return transforms.ToTensor()(pil_image)

# reformat dataset huggingface
def map_hf_2_lightning(examples):
    image = examples["image"]
    mask = examples["mask"]
    masked_image = examples["masked_image"]
    examples["image"] = [pil_to_tensor(img) for img in image]
    # examples["mask"] = [pil_to_tensor(m) for m in mask]
    examples["mask"] = [pil_to_tensor(m.convert("L")) for m in mask]
    examples["masked_image"] = [pil_to_tensor(mi) for mi in masked_image]
    return examples

def preprocess_image(image, mask, masked_image, size=256, random_crop=True):
        # image = Image.open(image_path)
        if size is not None and size > 0:
            rescaler = albumentations.SmallestMaxSize(max_size = size)
            # print(f"self.random_crop, {self.random_crop}")
            if not random_crop:
                cropper = albumentations.CenterCrop(height=size,width=size)
            else:
                cropper = albumentations.RandomCrop(height=size,width=size)
            preprocessor = albumentations.Compose(
                    [
                        rescaler, 
                        cropper
                    ],
                    additional_targets={"image": "image", "mask": "image", "masked_image": "image"}
                )
        else:
            # print("Existed")
            preprocessor = lambda **kwargs: kwargs
        # if not image.mode == "RGB":
        #     image = image.convert("RGB")
        image = image.permute(1,2,0).numpy()
        # normalize image to [0, 225]
        image = (image + 1.0) * 127.5
        mask = mask.permute(1,2,0).numpy()
        masked_image = masked_image.permute(1,2,0).numpy()
        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)
        masked_image = (masked_image + 1.0) * 127.5
        masked_image = np.array(masked_image).astype(np.uint8)
        # image = preprocessor(image=image)["image"]
        # mask = preprocessor(image=mask)["image"]
        # masked_image = preprocessor(image=masked_image)["image"]
        image = preprocessor(image=image, mask=mask, masked_image=masked_image)["image"]
        mask = preprocessor(image=image, mask=mask, masked_image=masked_image)["mask"]
        masked_image = preprocessor(image=image, mask=mask, masked_image=masked_image)["masked_image"]
        # print("a")
        image = (image/127.5 - 1.0).astype(np.float32)
        mask = (mask/127.5 - 1.0).astype(np.float32)
        # change shape mask image from (256,256,1) to (256,256)
        # mask = np.transpose(mask, (2, 0, 1))[0]
        mask = mask[:,:,0]
        # change shape mask image from (1,256,256) to (256,256)
        masked_image = (masked_image/127.5 - 1.0).astype(np.float32)
        return image, mask, masked_image

def data_collate_train(batch):
    
    # apply preprocess_image to each element in the batch list[dict[image, mask, masked_image]]
    examples = {}
    for key in batch[0].keys():
        examples[key] = []
    for element in batch:
        image, mask, masked_image = preprocess_image(element["image"], element["mask"], element["masked_image"])
        examples["image"].append(image)
        examples["mask"].append(mask)
        examples["masked_image"].append(masked_image)    
    # convert to tensor
    for key in examples.keys():
        examples[key] = np.array(examples[key])
        examples[key] = torch.tensor(examples[key])
    return examples

def data_collate_val(batch):
    
    # apply preprocess_image to each element in the batch list[dict[image, mask, masked_image]]
    examples = {}
    for key in batch[0].keys():
        examples[key] = []
    for element in batch:
        image, mask, masked_image = preprocess_image(element["image"], element["mask"], element["masked_image"], random_crop=False)
        examples["image"].append(image)
        examples["mask"].append(mask)
        examples["masked_image"].append(masked_image)    
    # convert to tensor
    for key in examples.keys():
        examples[key] = np.array(examples[key])
        examples[key] = torch.tensor(examples[key])
    return examples

class data_path_hf:
    def __init__(self, **kwargs):
        pass