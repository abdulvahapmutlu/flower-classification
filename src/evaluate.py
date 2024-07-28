import torch
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import get_data_loaders
from model import initialize_model

def evaluate_model(model, dataloaders, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    data_dir = "./data"
    num_classes = 17
    batch_size = 32
    feature_extract = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders, dataset_sizes, class_names = get_data_loaders(data_dir, batch_size)
    model_ft = initialize_model(num_classes, feature_extract, use_pretrained=False)
    model_ft.load_state_dict(torch.load('./models/flower_classifier.pth'))
    model_ft = model_ft.to(device)

    evaluate_model(model_ft, dataloaders, class_names)
