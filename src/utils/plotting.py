import math


def gridize(ncols=2, figsize=(16, 8), sharex=False, sharey=False, wspace=0.25, hspace=0.35,
            title_from_key=True, suptitle=None):
    """
    Decorator that turns a single-axes plotting function into a grid plotter over a 2-level dict.
    The wrapped function must accept: (ax, key: str, results_dict: Optional[dict], **kwargs).
    """

    def decorator(single_plot_fn):
        def wrapper(keys, *, results_dict=None, ncols_override=None, suptitle_override=None,
                    figsize_override=None, **plot_kwargs):
            if not keys:
                raise ValueError("No keys to plot.")

            ncols_eff = ncols_override or ncols
            nrows = math.ceil(len(keys) / ncols_eff)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols_eff, figsize=figsize_override or figsize,
                                    sharex=sharex, sharey=sharey)

            # Normalize axs to 2D array for consistent indexing
            if nrows == 1 and ncols_eff == 1:
                axs = [[axs]]
            elif nrows == 1:
                axs = [axs]
            elif ncols_eff == 1:
                axs = [[ax] for ax in axs]

            # Fill plots
            for idx, parent_key in enumerate(keys):
                r, c = divmod(idx, ncols_eff)
                ax = axs[r][c]
                single_plot_fn(ax=ax, key=parent_key, results_dict=results_dict, **plot_kwargs)
                if title_from_key and ax.get_title() == '':
                    ax.set_title(parent_key)

            # Hide unused axes (last row padding)
            total = nrows * ncols_eff
            for idx in range(len(keys), total):
                r, c = divmod(idx, ncols_eff)
                axs[r][c].set_visible(False)

            fig.tight_layout()
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
            if suptitle_override or suptitle:
                fig.suptitle(suptitle_override or suptitle)
                fig.tight_layout()
            return fig, axs

        return wrapper

    return decorator


import functools
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def with_axes(auto_show: bool = True, figsize: Tuple[float, float] = None):
    """
    Decorator for single-axes plot functions that accept `ax=None`.
    - Creates fig/ax if ax is None.
    - Sets the current axes to `ax` so `plt.*` works.
    - Calls plt.show() only if we created the figure and auto_show=True.

    The wrapped function must accept `ax=` in its signature.
    Returns (fig, ax) if we created them, otherwise returns the original function's return.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ax = kwargs.get("ax", None)
            created = ax is None
            if created:
                fig, ax = plt.subplots(figsize=kwargs.get("figsize", figsize))
                kwargs["ax"] = ax
            else:
                # If caller passed ax, try to recover the fig for consistency
                fig = ax.get_figure() if hasattr(ax, "get_figure") else None

            # Make sure plt.* APIs act on the correct axes
            if ax is not None:
                plt.sca(ax)

            out = fn(*args, **kwargs)

            if created and auto_show:
                plt.show()

            # If we created the figure, return it for convenience
            return (fig, ax) if created else out

        return wrapper

    return decorator


def plot_grid(plot_fn: callable, keys: list[str], results_dict: Optional[dict], *, ncols=2, figsize=(16, 8),
              suptitle=None, sharex=False, sharey=False, **kwargs):
    @gridize(ncols=ncols, figsize=figsize, suptitle=suptitle, sharex=sharex, sharey=sharey)
    def _wrapped(ax, key, results_dict, **kw):
        plot_fn(ax=ax, key=key, results_dict=results_dict, **kw)

    return _wrapped(keys=keys, results_dict=results_dict, **kwargs)
