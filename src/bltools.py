import matplotlib.pyplot as plt
def letter_subplots(axes=None, letters=None, xoffset=-0.1, yoffset=1.0, xlabel=None, ylabel=None, **kwargs):
    """Add letters to the corners of subplots. By default each axis is given an
    upper-case bold letter label.

    axes: list of pyplot ax objects.
    letters: list of strings to use as labels, default ["A", "B", "C", ...]
    xoffset, yoffset: positions of each label relative to plot frame (default
        -0.1,1.0 = upper left margin). Can also be a list of offsets, in which
        case it should be the same length as the number of axes.
    xlabel,ylabel: (optional) add label(s) to all the axes

    Other arguments will be passed to plt.annotate()

    Examples:
        >>> fig, axes = plt.subplots(1,3)
        >>> letter_subplots(axes, letters=['(a)', '(b)', '(c)'], fontweight='normal')

        >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
        >>> letter_subplots(fig.axes) # fig.axes is a list when axes is a 2x2 matrix
    """

    # handle single axes:
    if axes is None:
        axes = plt.gcf().axes
    try:
        iter(axes)
    except TypeError:
        axes = [axes]

    # set up letter defaults (and corresponding fontweight):
    fontweight = "bold"
    ulets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(axes)])
    llets = list('abcdefghijklmnopqrstuvwxyz'[:len(axes)])
    if letters is None or letters == "A":
        letters = ulets
    elif letters == "(a)":
        letters = [ "({})".format(lett) for lett in llets ]
        fontweight = "normal"
    elif letters == "(A)":
        letters = [ "({})".format(lett) for lett in ulets ]
        fontweight = "normal"
    elif letters == "lower" or letters == "lowercase" or letters == "a":
        letters = llets

    # make sure there are x and y offsets for each ax in axes:
    if isinstance(xoffset, (int, float)):
        xoffset = [xoffset]*len(axes)
    else:
        assert len(xoffset) == len(axes)
    if isinstance(yoffset, (int, float)):
        yoffset = [yoffset]*len(axes)
    else:
        assert len(yoffset) == len(axes)

    # defaults for annotate (kwargs is second so it can overwrite these defaults):
    my_defaults = dict(fontweight=fontweight, fontsize='large', ha="center",
                       va='center', xycoords='axes fraction', annotation_clip=False)
    kwargs = dict( list(my_defaults.items()) + list(kwargs.items()))

    list_txts = []
    for ax,lbl,xoff,yoff in zip(axes,letters,xoffset,yoffset):
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        t = ax.annotate(lbl, xy=(xoff,yoff), **kwargs)
        list_txts.append(t)
    return list_txts
