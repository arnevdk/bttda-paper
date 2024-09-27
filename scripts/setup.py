import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])
plt.rcParams.update({"font.size": 8})
plt.rcParams["lines.linewidth"] = 0.5


# Golden ratio to set aesthetic figure height
golden_ratio = (5**0.5 - 1) / 2


def set_size(width_in=6.3, fraction=1, aspect=golden_ratio, subplots=(1, 1)):
    # Figure width in inches
    fig_width_in = width_in
    # Figure height in inches
    fig_height_in = fig_width_in * aspect * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
