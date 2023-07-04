import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, size):
        """
        Create Replay buffer.

        Args:
            size (int): max number of samples to store in the buffer
                When the buffer overflows the old memories are dropped
        """
        self._buffer = []
        self._maxsize = size
        self._types = []

    def __len__(self):
        if self._buffer == []:
            return 0
        return len(self._buffer[0])

    def add(self, samples):
        """
        Add samples to the memory.

        Args:
            samples (tuple): tuple of input argument
                each element is expected to be a list of torch.tensor
        """
        if self._buffer == []:
            self._buffer = [
                np.asarray(samples[k]) if isinstance(samples[k], list)
                else samples[k].detach().cpu().numpy()
                for k in range(len(samples))
            ]
            self._types = [
                'list' if isinstance(samples[k], list) else 'tensor'
                for k in range(len(samples))
            ]
        else:
            if len(self._buffer[0]) + len(samples[0]) > self._maxsize:
                throw_away = np.random.randint(
                    0, len(self._buffer[0]),
                    (len(self._buffer[0]) + len(samples[0]) - self._maxsize,)
                )
                keep_inds = np.ones(len(self._buffer[0])).astype(bool)
                keep_inds[throw_away] = False
                self._buffer = [
                    self._buffer[k][keep_inds] for k in range(len(samples))
                ]
            for k in range(len(samples)):
                if isinstance(samples[k], list):
                    self._buffer[k] = np.concatenate((
                        self._buffer[k],
                        np.asarray(samples[k])
                    ))
                else:
                    self._buffer[k] = np.concatenate((
                        self._buffer[k],
                        samples[k].detach().cpu().numpy()
                    ))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        keep_inds = np.random.choice(
            len(self._buffer[0]), batch_size,
            replace=len(self._buffer[0]) < batch_size
        )
        return [
            self._buffer[k][keep_inds].tolist() if self._types[k] == 'list'
            else torch.from_numpy(self._buffer[k][keep_inds])
            for k in range(len(self._types))
        ]
