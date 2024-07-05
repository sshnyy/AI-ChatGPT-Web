import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def metric(prediction, label):
    _, max_indices = torch.max(prediction, 1)
    accuracy = (max_indices == label).sum().cpu().item() / max_indices.size()[0]

    prediction = torch.argmax(prediction, dim=-1).cpu().numpy()
    label = label.cpu().numpy()

    # 정밀도와 재현율 계산
    precision = precision_score(prediction, label, average="macro",zero_division=0)
    recall = recall_score(prediction, label, average="macro",zero_division=0)

    f1 = f1_score(
        prediction,
        label,
        average="macro",
    )
    return accuracy, precision, recall, f1, prediction, label
