"""
:Date        : 2025-12-08 19:59:32
:LastEditTime: 2025-12-23 23:10:00
:Description : 
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import traceback

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import ndimage, interpolate, signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.distance import cdist
from skimage import morphology, measure

def downsample_centerline_uniform(centerline, num_points=15):
    """
    Uniformly downsample 3D centerline
    
    Parameters:
        centerline: Original centerline, shape (N, 3)
        num_points: Target number of points

    Returns:
        Downsampled centerline, shape (num_points, 3)
    """
    if len(centerline) <= num_points:
        return centerline.copy()

    # Calculate cumulative chord length
    diffs = np.diff(centerline, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_length = np.zeros(len(centerline))
    cumulative_length[1:] = np.cumsum(segment_lengths)

    # Uniformly sample new parameter points on the curve
    # From 0 to total length, evenly take num_points points
    new_lengths = np.linspace(0, cumulative_length[-1], num_points)

    # Perform linear interpolation for X, Y, and Z separately
    interpolated = []
    for dim in range(3):  # Process X, Y, and Z dimensions separately
        interpolator = interp1d(cumulative_length, centerline[:, dim], 
                               kind='linear', fill_value='extrapolate')
        interpolated.append(interpolator(new_lengths))

    # Combine into new centerline
    downsampled = np.column_stack(interpolated)

    return downsampled

def stabilize_z_jitter(centerline_3d, method='spline'):
    """
    Quickly stabilize Z-axis jitter
    method: 'spline', 'savgol'
    """
    if len(centerline_3d) < 4:
        return centerline_3d

    z_original = centerline_3d[:, 2].copy()

    if method == 'spline':
        # Spline smoothing (preserve endpoints)
        t = np.arange(len(z_original))
        cs = CubicSpline(t, z_original)
        z_smooth = cs(t)

    elif method == 'savgol':
        # Savitzky-Golay filtering (retain peak features)
        window = min(11, len(z_original))
        if window % 2 == 0:
            window -= 1
        z_smooth = savgol_filter(z_original, window, 3)  # 3阶多项式

    else:
        # Moving average
        window = 5
        kernel = np.ones(window) / window
        z_smooth = np.convolve(z_original, kernel, mode='same')
        # Handle boundaries
        z_smooth[:window//2] = z_original[:window//2]
        z_smooth[-window//2:] = z_original[-window//2:]

    # Ensure it doesn't deviate too far from original values
    max_deviation = 1.5  # Maximum allowable deviation
    for i, z_original_dim in enumerate(z_original):
        if abs(z_smooth[i] - z_original_dim) > max_deviation:
            z_smooth[i] = 0.5 * z_original_dim + 0.5 * z_smooth[i]

    centerline_3d[:, 2] = z_smooth
    return centerline_3d


class Measurer3D:
    """
    3D Cilium Length Measurer (Optimized Version)
    Main optimizations:
    1. Use percentile thresholds to avoid non-specific staining interference
    2. MIP-guided 2D-3D hybrid measurement strategy
    3. Intelligent Z-axis positioning and smoothing
    4. Complete error handling and validation
    """

    def __init__(
        self,
        pixel_size_xy: float = 0.15,      # 微米/像素
        pixel_size_z: float = 0.5,        # 微米/切片
        top_percent: float = 0.0,         # 保留最亮的百分比像素
        min_cilium_length_3d: float = 0.5,# 最小有效纤毛长度(微米)
        subpixel_method: str = 'gaussian' # 'centroid'或'gaussian'
    ):
        """
        Initialize the measurer

        Args:
            pixel_size_xy: XY-plane pixel size (μm)
            pixel_size_z: Z-axis slice spacing (μm)
            top_percent: Percentage of brightest pixels to retain (default 3%)
            min_cilium_length_3d: Minimum valid cilium length (μm)
            subpixel_method: Subpixel localization method
        """
        self.pixel_size_xy = pixel_size_xy
        self.pixel_size_z = pixel_size_z
        self.top_percent = top_percent or 2.0
        self.min_cilium_length_3d = min_cilium_length_3d
        self.subpixel_method = subpixel_method

        # 验证参数
        assert 0.1 <= top_percent <= 20, "top_percent should be between 0.1-20%"
        assert subpixel_method in ['centroid', 'gaussian']

    def measure_from_3d_roi(
        self,
        roi_3d: np.ndarray
    ) -> Dict:
        """
        Measure cilium length from 3D ROI (main entry function)

        Args:
            roi_3d: 3D image region, shape (Z, Y, X)

        Returns:
            Dictionary containing measurement results
        """
        try:
            # Preprocessing (anisotropic filtering + normalization)
            processed = self._preprocess_volume(roi_3d)

            # Calculate Maximum Intensity Projection (MIP)
            mip_2d = np.max(processed, axis=0)

            # Extract 2D centerline from MIP (using percentile threshold)
            centerline_2d, binary_mip = self._extract_2d_centerline_from_mip(mip_2d)

            if len(centerline_2d) < 2:
                return self._create_empty_result()

            # Locate Z-axis position for each 2D point
            centerline_3d_raw = self._find_z_positions_3d(processed, centerline_2d)

            if len(centerline_3d_raw) < 2:
                return self._create_empty_result()

            # 3D centerline smoothing and optimization
            centerline_3d_smooth = stabilize_z_jitter(self._smooth_and_optimize_3d_centerline(centerline_3d_raw))

            # Calculate lengths
            length_3d = self._calculate_3d_length(centerline_3d_smooth)
            length_2d = self._calculate_2d_projected_length(centerline_3d_smooth)

            # Build result dictionary
            result = {
                'length_3d_um': length_3d,
                'length_2d_um': length_2d,
                'centerline_3d': centerline_3d_smooth,
                'centerline_2d': centerline_2d,
                'mip': mip_2d,
                'binary_mip': binary_mip
            }


            return result

        except Exception as e: # pylint: disable=broad-exception-caught
            print(repr(e))
            traceback.print_exc()
            return self._create_empty_result()

    # ==================== 核心算法步骤 ====================

    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        3D volume preprocessing: anisotropic Gaussian filtering + normalization
        """
        # Anisotropic filtering: less blurring in Z-axis than XY-plane
        sigma_z = 0.3   # Mild smoothing in Z direction
        sigma_xy = 0.7  # Stronger smoothing in XY directions

        smoothed = ndimage.gaussian_filter(
            volume.astype(np.float32),
            sigma=[sigma_z, sigma_xy, sigma_xy]
        )

        # Normalize to 0-1 range
        vmin, vmax = smoothed.min(), smoothed.max()
        if vmax > vmin:
            normalized = (smoothed - vmin) / (vmax - vmin)
        else:
            normalized = smoothed

        return normalized

    def _extract_2d_centerline_from_mip(
        self,
        mip: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 2D centerline from MIP (using percentile threshold)
        Returns: (centerline points (N,2), binary mask)
        """
        # Calculate percentile threshold (retain brightest top_percent% pixels)
        threshold_value = np.percentile(mip, 100 - self.top_percent)

        # Create binary mask
        binary = mip > threshold_value

        # Morphological cleanup
        binary = morphology.remove_small_objects(binary, min_size=20)
        binary = morphology.binary_closing(binary, morphology.disk(1))

        # Extract largest connected region
        labeled = measure.label(binary)
        if labeled.max() == 0:
            return np.array([]), binary

        regions = measure.regionprops(labeled, intensity_image=mip)
        largest_region = max(regions, key=lambda r: r.area)
        region_mask = labeled == largest_region.label

        # Extract 2D skeleton
        skeleton = morphology.skeletonize(region_mask, method='lee')

        # Get and order skeleton points
        y_coords, x_coords = np.where(skeleton)
        if len(x_coords) < 2:
            return np.array([]), binary

        points = np.column_stack([x_coords, y_coords])
        ordered_points = self._order_points_by_distance(points)

        # Interpolate centerline to obtain evenly distributed points
        if len(ordered_points) > 3:
            centerline_2d = self._interpolate_2d_centerline(ordered_points, num_points=50)
        else:
            centerline_2d = ordered_points

        return centerline_2d, binary

    def _find_z_positions_3d(
        self,
        volume_3d: np.ndarray,
        centerline_2d: np.ndarray
    ) -> np.ndarray:
        """
        Locate Z-axis position for each 2D center point
        Solves three key issues: multi-peak, noise, continuity
        """
        z_positions = []
        prev_z = None  # For continuity constraint

        for _, (x, y) in enumerate(centerline_2d):
            # Extract Z-axis intensity profile (with subpixel interpolation)
            z_profile = self._extract_z_profile_subpixel(volume_3d, x, y)

            # Intelligently select Z position
            z = self._find_optimal_z_position(
                z_profile,
                prev_z=prev_z
            )

            z_positions.append(z)
            prev_z = z  # Update previous point position

        # 组合成3D点
        centerline_3d = np.column_stack([
            centerline_2d[:, 0],  # X
            centerline_2d[:, 1],  # Y
            np.array(z_positions)  # Z
        ])

        return centerline_3d

    def _extract_z_profile_subpixel(
        self,
        volume: np.ndarray,
        x: float,
        y: float
    ) -> np.ndarray:
        """
        Extract Z-axis intensity profile at subpixel position
        """
        x_int, y_int = int(round(x)), int(round(y))

        # Simple bilinear interpolation to get subpixel intensity
        if (0 <= y_int < volume.shape[1] and
            0 <= x_int < volume.shape[2]):

            # Extract integer position profile
            base_profile = volume[:, y_int, x_int]

            # If close to integer position, return directly
            if abs(x - x_int) < 0.1 and abs(y - y_int) < 0.1:
                return base_profile

            # Otherwise use weighted average of adjacent points
            # x_frac, y_frac = x - x_int, y - y_int
            profiles = []
            weights = []

            for dx in [0, 1]:
                for dy in [0, 1]:
                    nx = min(volume.shape[2]-1, x_int + dx)
                    ny = min(volume.shape[1]-1, y_int + dy)
                    weight = (1 - abs(x - nx)) * (1 - abs(y - ny))

                    if weight > 0.1:
                        profiles.append(volume[:, ny, nx])
                        weights.append(weight)

            if profiles:
                weighted_profile = np.average(profiles, axis=0, weights=weights)
                return weighted_profile

        # Fallback
        return volume[:, int(round(y)), int(round(x))]

    def _find_optimal_z_position(
        self,
        z_profile: np.ndarray,
        prev_z: Optional[float] = None
    ) -> float:
        """
        Find optimal Z position (solves multi-peak, noise, continuity)
        """
        if len(z_profile) == 0:
            return prev_z if prev_z is not None else 0

        # Smooth profile
        smoothed = gaussian_filter1d(z_profile, sigma=1.0)

        # Method 1: Peak detection (for obvious peaks)
        peaks, properties = signal.find_peaks(
            smoothed,
            height=np.max(smoothed) * 0.3,
            distance=2,
            prominence=0.1
        )

        if len(peaks) == 1:
            # Single peak: subpixel refinement
            z = self._refine_peak_position(smoothed, peaks[0])

        elif len(peaks) > 1:
            # Multiple peaks: select based on continuity
            if prev_z is not None:
                # Select peak closest to previous point
                closest_idx = np.argmin(np.abs(peaks - prev_z))
                z = self._refine_peak_position(smoothed, peaks[closest_idx])
            else:
                # Select highest peak (starting point)
                highest_idx = np.argmax(properties['peak_heights'])
                z = self._refine_peak_position(smoothed, peaks[highest_idx])

        else:
            # No obvious peaks: use centroid method
            z_indices = np.arange(len(smoothed))
            total_intensity = np.sum(smoothed)

            if total_intensity > 0:
                z = np.sum(z_indices * smoothed) / total_intensity
            else:
                z = prev_z if prev_z is not None else len(z_profile) / 2

        # Continuity constraint: limit jump magnitude
        if prev_z is not None:
            max_jump = 2.0  # 最大允许Z跳跃
            if abs(z - prev_z) > max_jump:
                # Jump too large, apply smoothing
                z = 0.7 * prev_z + 0.3 * z

        # Ensure within valid range
        z = max(0, min(len(z_profile) - 1, z))

        return z

    def _refine_peak_position(
        self,
        profile: np.ndarray,
        peak_idx: int,
        window_radius: int = 2
    ) -> float:
        """
        Subpixel refinement of peak position
        """
        start = max(0, peak_idx - window_radius)
        end = min(len(profile), peak_idx + window_radius + 1)

        window = profile[start:end]
        indices = np.arange(start, end)

        if np.sum(window) == 0:
            return float(peak_idx)

        if self.subpixel_method == 'centroid':
            # Intensity-weighted centroid
            return np.sum(indices * window) / np.sum(window)

        elif self.subpixel_method == 'gaussian':
            # Gaussian fitting
            try:
                # Quadratic fitting after log transformation
                log_window = np.log(np.maximum(window, 1e-6))
                coeffs = np.polyfit(indices, log_window, 2)
                mu = -coeffs[1] / (2 * coeffs[0])
                return max(start, min(end-1, mu))
            except Exception as e: # pylint: disable=broad-exception-caught
                print(repr(e))
                # Fitting failed, fall back to centroid methods
                return np.sum(indices * window) / np.sum(window)

    def _smooth_and_optimize_3d_centerline(
        self,
        centerline: np.ndarray
    ) -> np.ndarray:
        """
        3D centerline smoothing and optimization
        """
        if len(centerline) < 4:
            return centerline

        # Calculate curvature-adaptive smoothing weights
        curvature_weights = self._calculate_3d_curvature_weights(centerline)

        # Apply adaptive smoothing to XYZ separately
        smoothed_centerline = centerline.copy()

        for dim in range(3):  # Process X,Y,Z separately
            original = centerline[:, dim]

            # Light smoothing
            smoothed = gaussian_filter1d(original, sigma=1.0)

            # Curvature-adaptive blending: preserve more original values where curvature is high
            adaptive_smoothed = np.zeros_like(original)
            for i, original_dim in enumerate(original):
                weight = 0.7 * (1 - curvature_weights[i])  # 曲率越大，平滑越少
                adaptive_smoothed[i] = weight * smoothed[i] + (1 - weight) * original_dim

            smoothed_centerline[:, dim] = adaptive_smoothed

        # Spline interpolation to obtain smooth uniform curve
        if len(smoothed_centerline) >= 4:
            final_centerline = self._spline_interpolate_3d(smoothed_centerline)
        else:
            final_centerline = smoothed_centerline

        return final_centerline

    def _calculate_3d_curvature_weights(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate curvature weights for 3D curve (for adaptive smoothing)
        """
        if len(points) < 3:
            return np.zeros(len(points))

        curvatures = np.zeros(len(points))

        for i in range(1, len(points)-1):
            p0, p1, p2 = points[i-1], points[i], points[i+1]

            # Calculate two vectors
            v1 = p1 - p0
            v2 = p2 - p1

            # Calculate angle (curvature proxy)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures[i] = angle

        # Endpoint handling
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]

        # Normalize to 0-1 range
        if np.max(curvatures) > 0:
            return curvatures / np.max(curvatures)
        else:
            return np.zeros_like(curvatures)

    def _spline_interpolate_3d(
        self,
        points: np.ndarray,
        num_points: int = 100
    ) -> np.ndarray:
        """
        3D spline interpolation
        """
        # Calculate cumulative chord length parameter
        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        t = np.zeros(len(points))
        t[1:] = np.cumsum(chords)
        t = t / t[-1]

        # Create new parameter points
        t_new = np.linspace(0, 1, num_points)

        # Spline interpolation for XYZ separately
        interp_coords = []
        for dim in range(3):
            if len(np.unique(points[:, dim])) >= 4:
                try:
                    spline = interpolate.CubicSpline(t, points[:, dim])
                    interp_coords.append(spline(t_new))
                except Exception as e: # pylint: disable=broad-exception-caught
                    print(repr(e))
                    interp_coords.append(np.interp(t_new, t, points[:, dim]))
            else:
                interp_coords.append(np.interp(t_new, t, points[:, dim]))

        return np.column_stack(interp_coords)

    def _order_points_by_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Order points by distance to form continuous path
        """
        if len(points) < 3:
            return points

        # 找到距离最远的两个点作为起点和终点
        dist_matrix = cdist(points, points)
        flat_idx = np.argmax(dist_matrix)
        start_idx = np.unravel_index(flat_idx, dist_matrix.shape)[0]

        # 最近邻排序
        ordered = [points[start_idx]]
        remaining = np.delete(points, start_idx, axis=0)

        while len(remaining) > 0:
            last_point = ordered[-1]
            distances = np.linalg.norm(remaining - last_point, axis=1)
            nearest_idx = np.argmin(distances)
            ordered.append(remaining[nearest_idx])
            remaining = np.delete(remaining, nearest_idx, axis=0)

        return np.array(ordered)

    def _interpolate_2d_centerline(
        self,
        points: np.ndarray,
        num_points: int = 50
    ) -> np.ndarray:
        """
        2D centerline interpolation to obtain evenly distributed points
        """
        if len(points) < 2:
            return points

        # 计算累积距离
        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        t = np.zeros(len(points))
        t[1:] = np.cumsum(chords)
        t = t / t[-1]

        # 创建新参数点
        t_new = np.linspace(0, 1, num_points)

        # 对X和Y分别插值
        interp_x = np.interp(t_new, t, points[:, 0])
        interp_y = np.interp(t_new, t, points[:, 1])

        return np.column_stack([interp_x, interp_y])

    # ==================== 长度计算 ====================

    def _calculate_3d_length(self, centerline: np.ndarray) -> float:
        """
        Calculate 3D centerline length
        """
        if len(centerline) < 2:
            return 0.0

        # Convert to physical coordinates
        scaled = centerline.copy()
        scaled[:, 0] *= self.pixel_size_xy  # X
        scaled[:, 1] *= self.pixel_size_xy  # Y
        scaled[:, 2] *= self.pixel_size_z   # Z

        # Calculate Euclidean distance between adjacent points
        diffs = np.diff(scaled, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))

        return float(np.sum(distances))

    def _calculate_2d_projected_length(self, centerline_3d: np.ndarray) -> float:
        """
        Calculate 2D projected length (XY plane)
        """
        if len(centerline_3d) < 2:
            return 0.0

        # Use XY coordinates
        xy_points = centerline_3d[:, :2].copy()
        xy_points[:, 0] *= self.pixel_size_xy  # X
        xy_points[:, 1] *= self.pixel_size_xy  # Y

        # Calculate distance between adjacent points
        diffs = np.diff(xy_points, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))

        return float(np.sum(distances))

    # ==================== Visualization ====================

    def visualize_3d_measurement(
        self,
        result: Dict,
        save_path: Optional[Path] = None
    ):
        """
        Complete visualization of measurement results
        """
        fig = plt.figure(figsize=(16, 12))

        # MIP and 2D centerline
        ax1 = fig.add_subplot(121)
        ax1.imshow(result['mip'], cmap='gray')
        formatter_xy = ticker.FuncFormatter(lambda x, pos: f'{x * self.pixel_size_xy:.1f}')
        ax1.xaxis.set_major_formatter(formatter_xy)
        ax1.yaxis.set_major_formatter(formatter_xy)

        # Set tick locator: every 1.5 micrometers
        ticks_every_um = 1.5 / self.pixel_size_xy
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(ticks_every_um))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(ticks_every_um))

        # Add axis labels with units
        ax1.set_xlabel('X (µm)')
        ax1.set_ylabel('Y (µm)')
        if len(result['centerline_2d']) > 0:
            ax1.plot(result['centerline_2d'][:, 0], result['centerline_2d'][:, 1],
                    'r-', linewidth=2, alpha=0.8)
        ax1.set_title(f'MIP 2D Length: {result["length_2d_um"]:.1f}μm')

        # 3D centerline (3D view)
        ax2 = fig.add_subplot(122, projection='3d')
        if len(result['centerline_3d']) > 0:
            centerline = downsample_centerline_uniform(
                result['centerline_3d'],
                max(3, int(result['length_3d_um']) + 1)
            )
            ax2.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2],
                    'r-', linewidth=3, alpha=0.8)
            ax2.scatter(centerline[0, 0], centerline[0, 1], centerline[0, 2],
                       c='g', s=100, marker='o')
            ax2.scatter(centerline[-1, 0], centerline[-1, 1], centerline[-1, 2],
                       c='b', s=100, marker='^')
            ax2.legend()
        formatter_xy = ticker.FuncFormatter(lambda x, pos: f'{x * self.pixel_size_xy:.1f}')
        formatter_z = ticker.FuncFormatter(lambda x, pos: f'{x * self.pixel_size_z:.1f}')
        ax2.xaxis.set_major_formatter(formatter_xy)
        ax2.yaxis.set_major_formatter(formatter_xy)
        ax2.zaxis.set_major_formatter(formatter_z)

        # Set Z-axis tick locator: every 0.5 micrometers
        ticks_every_um_z = 0.5 / self.pixel_size_z
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(ticks_every_um))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(ticks_every_um))
        ax2.zaxis.set_major_locator(ticker.MultipleLocator(ticks_every_um_z))

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f'3D Length: {result["length_3d_um"]:.1f}μm')

        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    def _create_empty_result(self) -> Dict:
        """
        Create empty result dictionary
        """
        return {
            'length_3d_um': 0.0,
            'length_2d_um': 0.0,
            'centerline_3d': np.zeros((0, 3)),
            'centerline_2d': np.zeros((0, 2)),
            'mip': None,
            'binary_mip': None,
        }
