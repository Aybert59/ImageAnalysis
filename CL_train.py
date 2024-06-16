# from great HF tuto at https://github.com/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb

from datasets import load_dataset, load_metric
import albumentations as A
import numpy as np
import torch
import evaluate
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer


# Step 1 : chose a model to train

print ('Starting')
model_checkpoint = "vincentclaes/mit-indoor-scenes"
#model_checkpoint = "/Users/olivier/ImageAnalysis/mit-indoor-scenes-finetuned-homepics-albumentations/checkpoint-100"
batch_size = 32 # batch size for training and evaluation
epochs = 200

print ('Loading Dataset')
# Step 2 : load dataset from homedir
#dataset = load_dataset("huggingface:jonathandinu/face-parsing") or just a name like "beans"
dataset = load_dataset("imagefolder", data_dir="/Volumes/photo/Datasets/pieces")

metric = evaluate.load("accuracy", trust_remote_code=True)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

#print (id2label)
#print (dataset["train"][20])
#print(dataset["train"].features)
#img = np.array(dataset["train"][20]["image"])
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cv2.imshow("w",img)
#cv2.waitKey(0)

print ('Prepare the transformer')
# Step 3 : prepare a transformer

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

#voir des exemples ici https://albumentations.ai/docs/getting_started/image_augmentation/
#transform = A.Compose([
#    A.RandomCrop(width=1024, height=1024),
#    A.HorizontalFlip(p=0.5),
#    A.RandomBrightnessContrast(p=0.2),
#    A.RandomGamma(p=0.2),
#
#])

train_transforms = A.Compose([
    A.RandomResizedCrop(size[1],size[0], scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), always_apply=False, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
    A.RandomBrightnessContrast (p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.Normalize(),
#    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = A.Compose([
    A.Resize(height=size[0], width=size[1]),
    A.Normalize(),
])

#def transforms(examples):
#    examples["pixel_values"] = [
#        transform(image=np.array(image))["image"] for image in examples["image"]
#    ]
#    return examples

def preprocess_train(examples):
    examples["pixel_values"] = [
        train_transforms(image=np.array(image))["image"] for image in examples["image"]
    ]
    return examples

def preprocess_val(examples):
    examples["pixel_values"] = [
        val_transforms(image=np.array(image))["image"] for image in examples["image"]
    ]
    return examples


#dataset.set_transform(transforms)
# split up training into training + validation
splits = dataset["train"].train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

print ('Prepare the model')
# Step 4 : prepare the model

num_labels = len(id2label)
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you'd like to fine-tune an already fine-tuned checkpoint
)

# Step 5 : prepare a trainer
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    output_dir=f"models/{model_name}-finetuned-homepics-albumentations",
    remove_unused_columns=False,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit = 5,
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        image = np.moveaxis(example["pixel_values"], source=2, destination=0)
        images.append(torch.from_numpy(image))
        labels.append(example["label"])
        
    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

print ("Let's go")
# Step 6 : train the model
trainer.train()

metrics = trainer.evaluate()
print(metrics)

