from datasets import DatasetDict, load_dataset
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score


def dataset_split(dataset):
    val_split_size=len(dataset["test"])/len(dataset["train"])
    train_val_split=dataset["train"].train_test_split(test_size=val_split_size, seed=42)
    dataset = DatasetDict(
        {"train": train_val_split["train"], "validation": train_val_split["test"], "test": dataset["test"]}
    )

    label_dict={id: label for id, label in enumerate(dataset["train"].features["label"].names)}

    return dataset,label_dict


## Augmentation 
def data_Augmentation(processor):

    mean = processor.image_mean
    std = processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=mean, std=std)

    train_transforms = Compose([
          RandomResizedCrop((size, size)),
          RandomHorizontalFlip(),
          ToTensor(),
          normalize,
    ])
    val_transforms = Compose([
            Resize((size, size)),
            ToTensor(),
            normalize,
    ])
    test_transforms = Compose([
            Resize((size, size)),
            ToTensor(),
            normalize,
    ])

    return train_transforms, val_transforms, test_transforms

# base on target to preprocess the dataset, and convert the image to tensor
def preprocess_dataset(sub_dataset,sub_dataset_name):
    match sub_dataset_name:
        case "train":
            pipline=train_transforms
        case "validation":
            pipline=val_transforms
        case "test":
            pipline=test_transforms
        case _:
            raise ValueError(f"Unknown sub_dataset: {sub_dataset}")
    sub_dataset["pixel_values"] = [pipline(img.convert("RGB")) for img in sub_dataset["image"]]
    return sub_dataset

# base on batch to collect the pixel_values and labels


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    
    return {"pixel_values": pixel_values, "labels": labels}

# custom compute metrics function to calculate accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


training_args = TrainingArguments(
    output_dir="vit_breastcancer_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    remove_unused_columns=False,
    
)

# train
def train(model,split_dataset,processor,training_args,):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()
    # 斷點繼續訓練
    # trainer.train(resume_from_checkpoint="../files/vit_breastcancer_model/checkpoint-120")
    trainer.save_model("vit_breastcancer_model/best")

def test(model,split_dataset,processor):
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=split_dataset["test"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    result=trainer.predict(split_dataset["test"])
    print(result)

if __name__ == "__main__":
    #load the dataset 
    dataset = load_dataset("emre570/breastcancer-ultrasound-images")

    # load vit encoder and preprocessor
    vit="google/vit-base-patch16-224-in21k"
    processor=ViTImageProcessor.from_pretrained(vit)
    split_dataset,label_dict=dataset_split(dataset)

    # data preprocessing and augmentation to tensor
    train_transforms, val_transforms, test_transforms = data_Augmentation(processor)
    for split_name, subset in split_dataset.items():
        subset.set_transform(
            lambda examples, s=split_name: preprocess_dataset(examples, s)
        )

    # 訓練用：從原始預訓練模型開始
    # model=ViTForImageClassification.from_pretrained(pretrained_model_name_or_path=vit,id2label=label_dict,ignore_mismatched_sizes=True)
    # 測試用：載入訓練好的最佳 weights
    model=ViTForImageClassification.from_pretrained("vit_breastcancer_model/best")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    #train(model,split_dataset,processor,training_args=training_args)
    test(model=model,split_dataset=split_dataset,processor=processor)











