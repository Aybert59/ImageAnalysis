# from great tuto at https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb
import torchvision
import os
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch

from pytorch_lightning import Trainer


# Step 1 : chose a model to train

print ('Starting')
model_checkpoint = "facebook/detr-resnet-50"
train_folder = '/Volumes/photo/Datasets/chaussures/train'
valid_folder = '/Volumes/photo/Datasets/chaussures/train'
batch_size = 8 # batch size for training and evaluation
epochs = 12
data_name = "result"
model_path = '/Users/olivier/ImageAnalysis/models/custom-detr-model'



print ('Creating dataset')

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, f"{data_name}.json" if train else f"{data_name}.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
    
processor = DetrImageProcessor.from_pretrained(model_checkpoint)

train_dataset = CocoDetection(img_folder=train_folder, processor=processor)
val_dataset = CocoDetection(img_folder=valid_folder, processor=processor, train=False)
training_steps = epochs * len(train_dataset)

print ('training_steps :', training_steps)
def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

batch = next(iter(train_dataloader))

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}
label2id = {v: k for k, v in id2label.items()}

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DetrForObjectDetection.from_pretrained(model_checkpoint,
                                                             revision="no_timm",
                                                             num_labels=len(id2label),
                                                             id2label=id2label,
                                                             label2id=label2id,
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
print (outputs.logits.shape)

trainer = Trainer(max_steps=training_steps, gradient_clip_val=0.1, devices=1, accelerator='mps')
trainer.fit(model)

model.model.save_pretrained(model_path)
#processor.config.to_json_file(f'{model_path}/preprocessor_config.json')
processor.save_pretrained(model_path)

#exit the program
exit(0)
