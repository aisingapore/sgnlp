from torch.optim.lr_scheduler import _LRScheduler
import warnings


class WarmupScheduler(_LRScheduler):
    """
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        n_warmup_steps (int): Number of steps for the warmup phase
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> scheduler = WarmupScheduler(optimizer, step_size=30, n_warmup_steps=100)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self, optimizer, step_size, n_warmup_steps, last_epoch=-1, verbose=False
    ):
        self.step_size = step_size
        self.n_warmup_steps = n_warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            base_lr * self._get_lr_factor()
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr * self._get_lr_factor()
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    def _get_lr_factor(self):
        return min(
            self._step_count ** (-0.5),
            self._step_count * (self.n_warmup_steps ** (-1.5)),
        )
