from typing import Optional
from torch.utils.data import Dataset, DataLoader


def get_data_loader(
    args,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    training: bool = True,
):
    if training:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )
        return train_loader, valid_loader
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
        )
        return test_loader
