"""
:Date        : 2025-12-02 10:17:58
:LastEditTime: 2025-12-23 23:09:51
:Description : 
"""
import numpy as np

def stretch_contrast(channel_data, lower_percentile, upper_percentile):
    """
    Stretch contrast
    """
    if channel_data.size == 0:
        return channel_data

    lower_val = np.percentile(channel_data, lower_percentile)
    upper_val = np.percentile(channel_data, upper_percentile)
    if upper_val <= lower_val:
        if channel_data.max() > 0:
            return channel_data / channel_data.max()
        return channel_data

    stretched = (channel_data - lower_val) / (upper_val - lower_val)
    return np.clip(stretched, 0, 1)


def merge_channels(channel_data_list, colors):
    """
    Merge channels. Outputs an RGB picture.
    """
    if not channel_data_list:
        return None

    height, width = channel_data_list[0].shape
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    for _, (channel_data, color) in enumerate(zip(channel_data_list, colors)):
        rgb_image[:, :, 0] += channel_data * color[0]  # R
        rgb_image[:, :, 1] += channel_data * color[1]  # B
        rgb_image[:, :, 2] += channel_data * color[2]  # G
    return np.clip(rgb_image, 0, 1)
