from src.dataset import load_lc25000, get_group_split, LungDataset, get_transforms, get_val_transforms
from torch.utils.data import DataLoader
from src.model import FusionModel
from src.train import train, train_kfold
from src.evaluate import evaluate
from src.config import Config

# ext_paths, ext_labels = load_external_dataset(Config.EXT_TEST_DIR)
# ext_dataset = LungDataset(ext_paths, ext_labels)
# ext_loader = DataLoader(ext_dataset, batch_size=Config.BATCH_SIZE)

# print("External Test:")
# evaluate(model, ext_loader)
def main():
    # Load dataset
    train_paths, train_labels, groups, test_paths, test_labels = load_lc25000(Config.TRAIN_DIR)

    # Split
    train_idx, val_idx, _ = get_group_split(train_paths, train_labels, groups)

    train_transforms = get_transforms()
    val_transforms = get_val_transforms()

    train_dataset = LungDataset(
        [train_paths[i] for i in train_idx],
        [train_labels[i] for i in train_idx],
        train_transforms
    )

    val_dataset = LungDataset(
        [train_paths[i] for i in val_idx],
        [train_labels[i] for i in val_idx],
        val_transforms
    )

    test_dataset = LungDataset(
        test_paths,
        test_labels,
        val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = FusionModel(num_classes=Config.NUM_CLASSES)

    # Train
    train(model, train_loader, val_loader)

    # Internal test
    print("Internal Test:")
    evaluate(model, test_loader)

    # # External test (MOST IMPORTANT STEP)
    # ext_paths, ext_labels = load_external_dataset(Config.EXT_TEST_DIR)

    # ext_dataset = LungDataset(ext_paths, ext_labels)
    # ext_loader = DataLoader(ext_dataset, batch_size=Config.BATCH_SIZE)

    # print("External Test:")
    # evaluate(model, ext_loader)


if __name__ == "__main__":
    main()