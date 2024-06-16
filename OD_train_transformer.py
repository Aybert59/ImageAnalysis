# from great tuto at https://danielvanstrien.xyz/huggingface/huggingface-datasets/transformers/2022/08/16/detr-object-detection.html
import torchvision
import os
from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, TrainingArguments, Trainer
import torch
import numpy as np


# Step 1 : chose a model to train

print ('Starting')
model_checkpoint = "facebook/detr-resnet-50"
train_folder = '/Volumes/photo/Datasets/chaussures/train'
valid_folder = '/Volumes/photo/Datasets/chaussures/train'
batch_size = 8 # batch size for training and evaluation
epochs = 74
data_name = "result"
model_path = '/Users/olivier/ImageAnalysis/models/custom-detr-model-2'



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


def transform(example_batch):
    images = example_batch["image"]
    ids_ = example_batch["image_id"]
    objects = example_batch["objects"]
    targets = [
        {"image_id": id_, "annotations": object_} for id_, object_ in zip(ids_, objects)
    ]
    return processor(images=images, annotations=targets, return_tensors="pt")


#train_dataset = train_dataset.with_transform(transform) COCO object has no with_transform method

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}
label2id = {v: k for k, v in id2label.items()}

model = DetrForObjectDetection.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    output_dir=f"models/{model_name}-finetuned-homepics",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit = 5,
    fp16=False,
    logging_steps=50,
    learning_rate=1e-4,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
)


print ("Let's go")
# Step 6 : train the model
trainer.train()





model.model.save_pretrained(model_path)
#processor.config.to_json_file(f'{model_path}/preprocessor_config.json')
processor.save_pretrained(model_path)

#exit the program
exit(0)
