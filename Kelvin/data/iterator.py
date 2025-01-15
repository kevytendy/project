from .dataset import Dataset
import torch
import numpy as np
import math

class Iterator:
    def __init__(self,
                 dataset,
                 batch_size,
                 sort_key=None,
                 reverse=False,
                 shuffle=False,
                 transpose=False,
                 square=False,
                 pad=None,
                 labels=None):
        """
        Initializes the Iterator object for batching and sampling data.

        Args:
            dataset: Dataset object containing the data.
            batch_size: Number of samples per batch.
            sort_key: Column to sort the data by, if any.
            reverse: Whether to reverse the sort order.
            shuffle: Whether to shuffle the data.
            transpose: Whether to transpose the batches.
            square: Whether to reshape the data into square matrices.
            pad: Padding value to use if data needs to be padded.
            labels: List of label columns, if applicable.
        """
        assert batch_size > 1, "Batch size must be greater than 1."
        self.batch_size = batch_size
        self.transpose = transpose
        self.labels = labels
        self.dataset = dataset
        self.shuffle = shuffle

        # Concatenate all columns and labels from the dataset
        self.concat()

        # Sort the data by the specified key, if provided
        if sort_key is not None:
            idx = self.dataset.columns.index(sort_key)
            sort_order = np.argsort(self.data[:, idx])
            if reverse:
                sort_order = sort_order[::-1]
            self.data = self.data[sort_order]
            if self.label is not None:
                self.label = self.label[sort_order]

        # Shuffle the data and labels, if applicable
        if self.shuffle:
            indices = np.random.permutation(len(self.data))
            self.data = self.data[indices]
            if self.label is not None:
                self.label = self.label[indices]

        # Reshape data into square matrices, if required
        if square:
            assert self.labels is None, "Square transformation does not support labeled data."
            assert pad is not None, "Pad value must be provided for square transformation."
            shape = int(np.ceil(np.sqrt(self.data.shape[1])))
            pad_num = shape * shape - self.data.shape[1]
            if pad_num != 0:
                padding = pad * np.ones([self.data.shape[0], pad_num])
                self.data = np.concatenate([self.data, padding], axis=1)
            self.data = self.data.reshape(-1, shape, shape)
            self.shape = shape

        # Create batches of data (and labels, if provided)
        self.batch_x = []
        self.batch_y = []

        num_batches = math.ceil(len(self.data) / self.batch_size)
        for i in range(num_batches):
            batch_data = self.data[i * self.batch_size: (i + 1) * self.batch_size]

            # Apply padding if necessary
            if pad is not None:
                max_len = max(row.shape[0] for row in batch_data)
                batch_data = np.array([
                    np.pad(row, (0, max_len - row.shape[0])) if row.shape[0] < max_len else row[:max_len]
                    for row in batch_data
                ])

            self.batch_x.append(batch_data)
            if self.label is not None:
                self.batch_y.append(self.label[i * self.batch_size: (i + 1) * self.batch_size])

        self.batch_x = np.asarray(self.batch_x, dtype=object)
        self.batch_y = np.asarray(self.batch_y, dtype=object)
        self.iter_idx = 0

    def concat(self):
        """
        Concatenates all feature columns into a single data array
        and all label columns (if provided) into a label array.
        """
        columns = [self.dataset.__dict__[col].convert() for col in self.dataset.columns if col not in (self.labels or [])]
        self.data = np.hstack(columns)  # Combine all feature columns at once

        # Pad rows to ensure consistent lengths
        max_length = max(row.shape[0] for row in self.data)
        self.data = np.array([
            np.pad(row, (0, max_length - row.shape[0])) if row.shape[0] < max_length else row[:max_length]
            for row in self.data
        ])

        # Combine label columns into a single array if provided
        if self.labels:
            labelcols = [self.dataset.__dict__[col].convert() for col in self.labels]
            self.label = np.hstack(labelcols)  # Combine all label columns at once
        else:
            self.label = None

    def sample(self, num_sample=None, label=None):
        """
        Samples data (and labels, if applicable) from the dataset.

        Args:
            num_sample: Number of samples to return.
            label: Label value to filter by, if applicable.

        Returns:
            Sampled data (and labels, if applicable) as torch tensors.
        """
        if num_sample is None:
            num_sample = self.batch_size

        if label is not None:
            assert self.label is not None, "Label column is not provided."
            label_indices = np.where(self.label == label)[0]
            if len(label_indices) == 0:
                raise ValueError(f"No data found for label: {label}")
            sampled_indices = np.random.choice(label_indices, size=num_sample, replace=False)
        else:
            sampled_indices = np.random.choice(len(self.data), size=num_sample, replace=False)

        sample_data = self.data[sampled_indices]
        sample_label = self.label[sampled_indices] if self.label is not None else None

        x = torch.from_numpy(sample_data.astype(np.float32))
        y = torch.from_numpy(sample_label.astype(np.float32)) if sample_label is not None else None

        return (x, y) if y is not None else x

    @classmethod
    def split(cls,
              batch_size,
              train=None,
              validation=None,
              test=None,
              labels=None,
              shuffle=False,
              sort_key=None,
              reverse=False,
              transpose=False,
              square=False,
              pad=None):
        """
        Splits the dataset into train, validation, and test iterators.

        Args:
            batch_size: Batch size for all splits.
            train: Training dataset.
            validation: Validation dataset.
            test: Test dataset.
            labels: List of label columns, if applicable.
            shuffle: Whether to shuffle the data.
            sort_key: Column to sort the data by, if any.
            reverse: Whether to reverse the sort order.
            transpose: Whether to transpose the batches.
            square: Whether to reshape the data into square matrices.
            pad: Padding value to use if data needs to be padded.

        Returns:
            Tuple of iterators for the provided splits.
        """
        iterator_args = {'batch_size': batch_size,
                         'shuffle': shuffle,
                         'sort_key': sort_key,
                         'reverse': reverse,
                         'transpose': transpose,
                         'square': square,
                         'pad': pad}
        train_it = None if train is None else cls(dataset=train, labels=labels, **iterator_args)
        valid_it = None if validation is None else cls(dataset=validation, labels=labels, **iterator_args)
        test_it = None if test is None else cls(dataset=test, labels=labels, **iterator_args)

        iterators = tuple(it for it in (train_it, valid_it, test_it) if it is not None)
        return iterators

    def __len__(self):
        """
        Returns the number of batches.
        """
        return len(self.batch_x)

    # Inside the `Iterator` class

    def __iter__(self):
        """
        Resets the iterator and returns itself.
        """
        self.iter_idx = 0  # Reset the iteration index at the start of an epoch
        return self

    def __next__(self):
        """
        Returns the next batch of data (and labels, if applicable).
        """
        if len(self.batch_x) == 0:
            raise StopIteration("No data batches available.")
        if self.iter_idx == len(self.batch_x):
            self.iter_idx = 0  # Reset for the next epoch to avoid conflict
            raise StopIteration(f"All {len(self.batch_x)} batches have been processed for this epoch.")

        # Fetch the current batch
        x = self.batch_x[self.iter_idx]
        y = self.batch_y[self.iter_idx] if self.labels and self.batch_y is not None else None

        # Handle transposition
        if self.transpose:
            if x.ndim == 2:
                x = x.T
            if y is not None and y.ndim == 2:
                y = y.T

        # Convert to PyTorch tensors
        x = torch.from_numpy(x.astype(np.float32))
        if y is not None:
            y = torch.from_numpy(y.astype(np.float32))

        # Increment the iterator index
        self.iter_idx += 1

        return (x, y) if y is not None else x

    def reset_epoch(self):
        """
        Manually reset the epoch, if needed.
        Useful when running multiple training loops or manually managing iterations.
        """
        self.iter_idx = 0
