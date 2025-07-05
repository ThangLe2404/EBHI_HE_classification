# Mount the drive into google colab
import io
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from imgaug import augmenters as iaa
import timm
warnings.filterwarnings("ignore")
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from thop import profile
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_score, 
    recall_score, 
    f1_score
)
from torchinfo import summary
from sklearn.preprocessing import label_binarize
CLASS_NAMES = ['Adenocarcinoma','High-grade IN',  'Low-grade IN', 'Normal', 'Polyp', 'Serrated']

def compute_class_accuracies(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    correct = np.diag(cm)
    total = cm.sum(axis=1)  # total true instances per class

    acc_per_class = correct / total
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(acc_per_class))]

    print("Per-Class Accuracy:")
    for name, acc in zip(class_names, acc_per_class):
        print(f"{name}: {acc:.4f}")

    return dict(zip(class_names, acc_per_class))

def plot_confusion_matrices(y_true, y_pred, num_classes, class_names=None, model_name='model'):
    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1, keepdims=True)

    # Fallback if no class names provided
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    plt.rcParams.update({'font.size': 14})  # You can change 14 to 16, 18, etc.
    # Raw Count Matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    raw_path = f"{model_name}_confmat_raw.png"
    plt.savefig(raw_path)
    plt.close()
    print(f"Saved raw confusion matrix to {raw_path}")
    plt.rcParams.update({'font.size': 14})  # You can change 14 to 16, 18, etc.
    # Normalized Matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    norm_path = f"{model_name}_confmat_normalized.png"
    plt.savefig(norm_path)
    plt.close()
    print(f"Saved normalized confusion matrix to {norm_path}")

def calculate_one_vs_rest_auc(probs, targets, num_classes, class_names=None):
    """
    Compute and print AUC for each class (one-vs-rest).
    """
    y_true = label_binarize(targets.numpy(), classes=list(range(num_classes)))
    y_score = probs.numpy()

    aucs = {}
    for i in range(num_classes):
        auc_i = roc_auc_score(y_true[:, i], y_score[:, i])
        class_label = class_names[i] if class_names else f"Class {i}"
        aucs[class_label] = auc_i
        print(f"[One-vs-Rest AUC] {class_label}: {auc_i:.4f}")
    return aucs

def plot_confusion_matrix(cm, class_names, model_name):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    path = f"{model_name}_confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved to {path}")

def plot_one_vs_rest_auc(probs, targets, num_classes, model_name, class_names=None):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np

    targets_one_hot = label_binarize(targets.numpy(), classes=list(range(num_classes)))
    probs_np = probs.numpy()

    plt.figure(figsize=(8, 6))

    plt.rcParams.update({'font.size': 21})  # You can change 14 to 16, 18, etc.
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(targets_one_hot[:, i], probs_np[:, i])
        auc_score = auc(fpr, tpr)
        label = class_names[i] if class_names else f"Class {i}"
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC Curves (Per Class)')
    plt.legend(loc="lower right", fontsize='medium')
    plt.tight_layout()
    save_path = f"{model_name}_one_vs_rest_auc.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved one-vs-rest AUC plot to {save_path}")

def plot_macro_micro_auc_curve(probs, targets, num_classes, model_name, class_names=None):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    import numpy as np

    targets_one_hot = label_binarize(targets.numpy(), classes=list(range(num_classes)))
    probs_np = probs.numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], probs_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(targets_one_hot.ravel(), probs_np.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.rcParams.update({'font.size': 21})  # You can change 14 to 16, 18, etc.
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], linestyle=':', linewidth=2, label=f"Micro-average (AUC = {roc_auc['micro']:.2f})")
    plt.plot(fpr["macro"], tpr["macro"], linestyle='-', linewidth=2, label=f"Macro-average (AUC = {roc_auc['macro']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro vs Macro ROC-AUC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_path = f"{model_name}_auc.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved micro/macro AUC plot to {save_path}")


IMAGE_SIZE = 224
models_type = {
    'resnet18': 'resnet18.a1_in1k',
    'resnet34': 'resnet34.a1_in1k',
    'resnet50': 'resnet50.a1_in1k',
    'swinv2_t_w8_256': 'swinv2_tiny_window8_256.ms_in1k',
    'swinv2_t_w16_256': 'swinv2_tiny_window16_256.ms_in1k',
    'swinv2_s_w8_256': 'swinv2_small_window8_256.ms_in1k',
    'swinv2_s_w16_256': 'swinv2_small_window16_256.ms_in1k',
    }

weight_paths = {
    'resnet18': ['./resnet18-epoch=8-val_loss=0.27-val_acc=0.91.ckpt'],
    'resnet34': [#'./resnet34-epoch=8-val_loss=0.35-val_acc=0.91.ckpt', 
                 './resnet34-epoch=11-val_loss=0.36-val_acc=0.88.ckpt'],# './resnet34-epoch=11-val_loss=0.36-val_acc=0.88.ckpt'],
    'resnet50': [#'./resnet50-epoch=29-val_loss=0.37-val_acc=0.90.ckpt', 
                 './resnet50-epoch=29-val_loss=0.37-val_acc=0.90.ckpt'],
    'swinv2_t_w8_256': ['./swinv2_t_w8_256-epoch=66-val_loss=0.41-val_acc=0.88.ckpt'] ,#,'./swinv2_t_w8_256-epoch=30-val_loss=0.42-val_acc=0.87.ckpt'],
    #'swinv2_t_w16_256': ['./swinv2_t_w16_256-epoch=46-val_loss=0.45-val_acc=0.89.ckpt','./swinv2_t_w16_256-epoch=34-val_loss=0.42-val_acc=0.88.ckpt'],
    'swinv2_s_w8_256': [#'./swinv2_s_w8_256-epoch=86-val_loss=0.47-val_acc=0.89.ckpt', './swinv2_s_w8_256-epoch=35-val_loss=0.38-val_acc=0.87.ckpt', 
                        './swinv2_s_w8_256-epoch=31-val_loss=0.47-val_acc=0.86.ckpt'
                        ]
    #'swinv2_s_w16_256': ['./swinv2_s_w8_256-epoch=86-val_loss=0.47-val_acc=0.89.ckpt'],
    }


optimizers = {"adam": Adam, "sgd": SGD}




# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class Classifier(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        model_name,
        train_path,
        val_path,
        test_path=None,
        optimizer="adam",
        lr=1e-4,
        batch_size=16,
        transfer=True,
        tune_fc_only=True,
        weight_idx=0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.lr = lr
        self.batch_size = batch_size
        self.layers = 2
        self.optimizer = optimizers[optimizer]
        # instantiate loss criterion
        self.weight_idx = weight_idx
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        )
        # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        self.model_name = model_name
        self.image_size = 256 if '256' in self.model_name else 224 
        print(self.image_size) 
        # Using a pretrained ResNet backbone
        #self.model = self.swin_models[swin_version](pretrained=transfer)
        self.model = timm.create_model(model_name=models_type[model_name], pretrained=transfer)#, in_chans=3)
        # Replace old FC layer with Identity so we can train our own
        if 'swin' in self.model_name:
            linear_size = self.model.head.in_features
            self.model.head.fc = nn.Linear(in_features=linear_size, out_features=6, bias=True)
            print(self.model)

        else:
            #x = torch.randn(1, 3, 224, 224)
            #with torch.no_grad():
            #    feats = self.model.forward_features(x)
        #    linear_size = feats.shape[1]  # torch.Size([1, 2048])
                        #print(self.model)                  
             self.model.reset_classifier(num_classes=num_classes)  # Just changes output size, keeps it linear
        # replace final layer for fine tuning
              #self.model = nn.Sequential(*list(self.model.children())[:-3])
        # nn.Linear(linear_size, num_classes)
        #print(self.model)
        #if tune_fc_only:  # option to only tune the fully-connected layers
        #    for child in list(self.resnet_model.children())[:-1]:
        #        for param in child.parameters():
        #            param.requires_grad = False
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        #self.finetune()
        self.top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.top2 = MulticlassAccuracy(num_classes=num_classes, top_k=2)
        self.top3 = MulticlassAccuracy(num_classes=num_classes, top_k=3)
        #self.top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        #self.micro_auroc = MulticlassAUROC(num_classes=num_classes, average='micro')
        self.marcro_auroc = MulticlassAUROC(num_classes=num_classes, average='macro')


        self.confmat = MulticlassConfusionMatrix(num_classes=num_classes)

        self.preds = []
        self.probs = []
        self.targets = []

        self.test_stats = {}

    def on_test_start(self):
        # Estimate FLOPs and Params
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(next(self.model.parameters()).device)
        flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)

        self.test_stats["FLOPs"] = flops
        self.test_stats["Params"] = params

        print(f"[INFO] Model FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"[INFO] Model Parameters: {params / 1e6:.2f} Million")
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.preds)
        probs = torch.cat(self.probs)
        targets = torch.cat(self.targets)

        top1 = self.top1.compute()
        top2 = self.top2.compute()
        top3 = self.top3.compute()
        #top5 = self.top5.compute()
        print(preds, np.eye(6)[targets.numpy()])
        micro_auroc_score = roc_auc_score(targets.numpy(), probs.numpy(), multi_class='ovr', average='micro')
        macro_auroc_score = self.marcro_auroc(probs, targets)
        #print(self.confmat.device, probs.device) 
        self.confmat = self.confmat.to("cpu")
        confmat_result = self.confmat(probs.argmax(dim=1), targets)

        self.confusion_matrix = confmat_result.numpy()
        self.test_preds = preds
        self.test_targets = targets
        self.test_probs = probs
        self.micro_auroc_score = micro_auroc_score# micro_auroc_score.item()
        self.macro_auroc_score = macro_auroc_score.item()
        self.ovr_per_class = calculate_one_vs_rest_auc(
            probs=probs,
            targets=targets,
            num_classes=6,
            class_names=CLASS_NAMES
        )
        precision = precision_score(targets.numpy(), probs.argmax(dim=1).numpy(), average='macro')
        recall = recall_score(targets.numpy(), probs.argmax(dim=1).numpy(), average='macro')
        f1 = f1_score(targets.numpy(), probs.argmax(dim=1).numpy(), average='macro')

        precision_pc = precision_score(targets.numpy(), probs.argmax(dim=1).numpy(), average=None)
        recall_pc= recall_score(targets.numpy(), probs.argmax(dim=1).numpy(), average=None)
        f1_pc = f1_score(targets.numpy(), probs.argmax(dim=1).numpy(), average=None)

        buffer = io.StringIO()
        sys.stdout = buffer  # Redirect stdout
        accs = compute_class_accuracies(targets.numpy(), preds.argmax(dim=1).numpy(), CLASS_NAMES)
        summary(self.model, input_size=(1, 3, self.image_size, self.image_size), depth=4, 
        col_names=("input_size", "output_size", "num_params", "params_percent"))
        sys.stdout = sys.__stdout__
        # Save to file
        with open(f"{self.model_name}_{self.weight_idx}.txt", "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"FLOPs: {self.test_stats['FLOPs'] / 1e9:.2f} GFLOPs\n")
            f.write(f"Params: {self.test_stats['Params'] / 1e6:.2f} Million\n")
            f.write(f"Top-1 Accuracy: {top1:.4f}\n")
            f.write(f"Top-2 Accuracy: {top2:.4f}\n")
            f.write(f"Top-3 Accuracy: {top3:.4f}\n")
            f.write(f"Accuracy per class\n")
            for name, acc in accs.items():
                f.write(f"{name}: {acc:.4f}\n")
            
            f.write(f"F1: {f1:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write("Per-Class F1, Recall, Precision:\n")
            for i, name in enumerate(CLASS_NAMES):
                f.write(f"{name}: Precision={precision_pc[i]:.3f}, Recall={recall_pc[i]:.3f}, F1={f1_pc[i]:.3f}\n")
           #f.write(f"Top-5 Accuracy: {top5:.4f}\n")
            f.write(f"Micro AUC: {self.micro_auroc_score:.4f}\n")
            f.write(f"Macro AUC: {self.macro_auroc_score:.4f}\n")
            f.write("\nOne-vs-Rest AUCs:\n")
            for cls, auc_val in self.ovr_per_class.items():
                f.write(f"{cls}: {auc_val:.4f}\n")
            f.write(buffer.getvalue())


        print(f"Test summary written to {self.model_name}.txt")
        plot_macro_micro_auc_curve(probs, targets, 6, f"{self.model_name}_{self.weight_idx}", class_names=CLASS_NAMES)
        plot_one_vs_rest_auc(probs, targets, 6, f"{self.model_name}_{self.weight_idx}", class_names=CLASS_NAMES)
        plot_confusion_matrices( targets, probs.argmax(dim=1).numpy(), 6, class_names=CLASS_NAMES, model_name=f"{self.model_name}_{self.weight_idx}")

    
    def finetune(self):

        for param in list(self.model.children())[:-1]:
            for p in param.parameters():
                p.requires_grad = False
    
        for param in list(self.model.layers._modules['3'].blocks):
            for p in param.parameters():
                p.requires_grad = True

        if self.layers != 0:
            for param in list(self.model.layers._modules['2'].blocks)[-self.layers:]:
                for p in param.parameters():
                    p.requires_grad = True


    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr = self.lr, 
            weight_decay = 1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = 'max', 
            factor = 0.2, 
            patience = 50, 
            verbose = True)

        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_acc' 
       }
    
    #return  
    #return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch
        preds = self(x)
        #print(preds.shape, y.shape)
        #print(preds.unsqueeze(0), preds.unsqueeze(0).shape)
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc

    def _dataloader(self, data_path, shuffle=False):
        img_size = self.image_size
        # values here are specific to pneumonia dataset and should be updated for custom data
        tramsform = None
        if "train" in data_path:
          transform = transforms.Compose([
                np.asarray,
                    iaa.Sequential([
                    iaa.Resize({"height": img_size, "width": img_size})
                    ]).augment_image,
                np.copy,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225])]
          )
        else:
          transform = transforms.Compose([
                np.asarray,
                  iaa.Sequential([
                    iaa.Resize({"height": img_size, "width": img_size})
                  ]).augment_image,
                np.copy,

                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225])
            ]
          )

        img_folder = ImageFolder(data_path, transform=transform)

        return DataLoader(img_folder, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        return self._dataloader(self.train_path, shuffle=True)

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def val_dataloader(self):
        return self._dataloader(self.val_path)

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        return self._dataloader(self.test_path)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        
        #preds = torch.argmax(probs, dim=1)
        
        self.preds.append(logits.detach().cpu())
        self.probs.append(probs.detach().cpu())
        self.targets.append(y.detach().cpu())

        self.top1.update(probs, y)#, prog_bar=True)
        self.top2.update(probs, y)
        self.top3.update(probs, y)
        #self.log("test_top5", self.top5(logits, y))
        return {}


def main( 
    weight_idx,
    model_name, 
    num_classes=6,
    train_path='./new_train',
    val_path='./val',
    test_path='./test',
    optimizer='adam',
    lr=1e-3,
    batch_size=32,
    transfer=True,
    tune_fc_only=False,
    num_epochs = 30,
    save_path = '.',
    mixed_precision = False):


    # # Instantiate Model
    model = Classifier.load_from_checkpoint(weight_paths[model_name][weight_idx],
        num_classes=num_classes,
        model_name=model_name,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        optimizer=optimizer,
        lr=lr,
        batch_size=batch_size,
        transfer=transfer,
        tune_fc_only=tune_fc_only,
        weight_idx=weight_idx
    )
    '''
    save_path = save_path if save_path is not None else "./models"
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename=model_name + "-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    )
    
    stopping_callback=pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=40)
    # Instantiate lightning trainer and train model
    trainer_args = {
        "accelerator": "cuda",
        "max_epochs": num_epochs,
        "callbacks": [checkpoint_callback, stopping_callback],
        "precision": 16 if mixed_precision else 32,
    }
    '''
    trainer = pl.Trainer()#**trainer_args, )
    
    #trainer.fit(model)

    #if test_path:
    trainer.test(model)
    # Save trained model weights
    #torch.save(trainer.model.resnet_model.state_dict(), save_path + "/trained_model.pt")

    return model

#
#

# resnet 30 epoch
for model_name in weight_paths.keys():
    for weight_idx in range(len(weight_paths[model_name])):
        model = main(weight_idx, model_name, num_epochs=100, batch_size=1)
     
