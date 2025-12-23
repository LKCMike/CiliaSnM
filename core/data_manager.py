"""
:Date        : 2025-11-22 10:17:58
:LastEditTime: 2025-12-23 23:10:09
:Description : 
"""
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import numpy as np
import tifffile
from aicsimageio.aics_image import AICSImage


class DataManager:
    """
    Image & metadata processing and management
    """
    def __init__(self):
        self.current_path: str = ''
        self.raw_data = None
        self.channel_names = []
        self.max_projection = None
        self.aics_image = None

    def load_image(self, file_path: str):
        """
        Load OME-TIFF File
        """
        try:
            self.current_path = file_path
            self.aics_image = AICSImage(file_path)

            # 获取数据，维度顺序为TCZYX
            self.raw_data = self.aics_image.get_image_data("TCZYX")
            self.channel_names = self.aics_image.channel_names

            # 计算最大投影
            self.compute_max_projection()

            return True
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Fail to load OME-TIFF image file: {e}")
            return False

    def compute_max_projection(self):
        """
        Z-stack MIP
        """
        if self.raw_data is None:
            return None

        # Channel data structure: (T, C, Z, Y, X)
        # T - Time sequence. Which we don't have
        # C - Channel. We merge all channels together.
        self.max_projection = np.max(self.raw_data[0, :, :, :, :], axis=1)
        return self.max_projection

    def get_channel_count(self):
        """
        How many channels do we have
        """
        return len(self.channel_names)

    def get_channel_wavelengths(self) -> List[Optional[float]]:
        """
        Emission Wavelength of each channel.
        We use these information for coloring the MIP merged preview.
        """
        wavelengths = []
        try:
            with tifffile.TiffFile(self.current_path) as tif:
                # OME Metada
                ome_xml : str = tif.ome_metadata
                # Parse XML with element tree.
                root = ET.fromstring(ome_xml)
                channels = root.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel')

                # Transform string to float.
                for channel in channels:
                    wavelength = channel.get('EmissionWavelength')
                    if wavelength:
                        wavelengths.append(float(wavelength))

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Exception occured while extracting Emission Wavelength: {e}")
            wavelengths = []

        return wavelengths

    @staticmethod
    def wavelength_to_rgb(wavelength_nm: Optional[float]) -> Tuple[float, float, float]:
        """
        Map Emission Wavelength(nm) to RGB Normalized channels
        """
        if wavelength_nm is None:
            return (0.8, 0.8, 0.8)  # Unknown? Gray.

        if wavelength_nm < 410:
            return (0.0, 0.0, 1.0)   # Purple
        elif 410 <= wavelength_nm < 470:
            return (0.0, 0.0, 1.0)   # Blue (DAPI, for example)
        elif 470 <= wavelength_nm < 480:
            return (0.0, 1.0, 1.0)   # Cyan (CFP)
        elif 480 <= wavelength_nm < 520:
            return (0.0, 1.0, 0.0)   # Green (FITC, GFP, etc)
        elif 520 <= wavelength_nm < 580:
            return (1.0, 1.0, 0.0)   # Yellow (YFP)
        else:
            return (1.0, 0.0, 0.0)   # Red (TRITC, AF594)

    def get_channel_colors(self) -> List[Tuple[float, float, float]]:
        """
        Color definitions in one list for all channels
        """
        wavelengths = self.get_channel_wavelengths()
        return [self.wavelength_to_rgb(wl) for wl in wavelengths]
