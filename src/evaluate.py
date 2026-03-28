import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from tqdm import tqdm

def evaluate(model, loader):
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():

        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(next(model.parameters()).device)
            out = model(x)

            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

    try:
        print("ROC-AUC:", roc_auc_score(y_true, y_prob, multi_class='ovr'))
    except:
        print("ROC-AUC failed")