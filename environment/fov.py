"""
Camera view projection module for projecting camera frustum to ground plane.

This module provides functionality to project a camera's view frustum onto a ground plane
given intrinsic and extrinsic camera parameters. Uses only numpy for computations.
"""

import numpy as np
from typing import Tuple, Optional


class CameraProjection:
    """Projects camera frustum to ground plane using intrinsic and extrinsic parameters."""

    def __init__(
        self,
        intrinsic_matrix: np.ndarray,
        extrinsic_matrix: np.ndarray,
        near_plane: float = 0.1,
        far_plane: float = 10.0,
    ):
        """
        Initialize camera projection with intrinsic and extrinsic matrices.

        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix (K matrix)
                            [[fx,  0, cx],
                             [ 0, fy, cy],
                             [ 0,  0,  1]]
            extrinsic_matrix: 4x4 camera extrinsic matrix (world to camera transformation)
                            [[R, t],
                             [0, 1]]
                            where R is 3x3 rotation and t is 3x1 translation
            near_plane: Distance to near clipping plane (default: 0.1)
            far_plane: Distance to far clipping plane (default: 10.0)
        """
        self.K = np.asarray(intrinsic_matrix, dtype=np.float64)
        self.T_cam_world = np.asarray(extrinsic_matrix, dtype=np.float64)
        
        # Extract rotation and translation
        self.R = self.T_cam_world[:3, :3]
        self.t = self.T_cam_world[:3, 3]
        
        # Camera to world transformation (inverse)
        self.R_inv = self.R.T
        self.t_inv = -self.R_inv @ self.t
        
        self.near = near_plane
        self.far = far_plane

    def get_frustum_corners_camera_frame(self) -> np.ndarray:
        """
        Get the 8 corners of the frustum in camera frame coordinates.

        Returns:
            Array of shape (8, 3) with normalized frustum corners in camera frame.
            Order: 4 corners at near plane, 4 at far plane.
        """
        # Get field of view from intrinsic matrix
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Image plane coordinates at unit depth (z=1)
        # These correspond to the principal point and edges
        width = cx * 2  # approximate image width
        height = cy * 2  # approximate image height

        # Corners in normalized image coordinates (before focal length scaling)
        corners_img = np.array([
            [-cx, -cy],      # top-left
            [width - cx, -cy],  # top-right
            [width - cx, height - cy],  # bottom-right
            [-cx, height - cy],  # bottom-left
        ])

        # Convert image coordinates to camera coordinates using inverse of K
        K_inv = np.linalg.inv(self.K)
        
        # Create homogeneous coordinates for image plane points at z=1
        corners_3d_near = []
        corners_3d_far = []

        for corner_2d in corners_img:
            # Unproject using inverse intrinsic matrix
            point_h = np.array([corner_2d[0], corner_2d[1], 1.0])
            ray_direction = K_inv @ point_h
            ray_direction = ray_direction / ray_direction[2]  # normalize by z

            # Scale by near and far plane distances
            corners_3d_near.append(ray_direction * self.near)
            corners_3d_far.append(ray_direction * self.far)

        # Stack near and far corners
        corners = np.vstack([corners_3d_near, corners_3d_far])
        return corners

    def transform_points_to_world(self, points_cam: np.ndarray) -> np.ndarray:
        """
        Transform points from camera frame to world frame.

        Args:
            points_cam: Array of shape (N, 3) in camera frame coordinates

        Returns:
            Array of shape (N, 3) in world frame coordinates
        """
        # X_world = R^T @ X_cam + t_inv
        points_world = points_cam @ self.R_inv.T + self.t_inv
        return points_world

    def project_to_ground_plane(
        self, points_3d: np.ndarray, ground_height: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Project 3D points onto ground plane using ray casting.

        Args:
            points_3d: Array of shape (N, 3) in world coordinates
            ground_height: Height of ground plane (default: 0.0)

        Returns:
            Array of shape (N, 2) with (x, y) coordinates on ground plane,
            or None if projection fails
        """
        # Project each point onto the ground plane
        # For a ray from camera center through a point, find intersection with ground plane

        # Camera center in world coordinates
        cam_center = self.t_inv

        # For each point, create a ray and find intersection with ground plane
        projected_points = []

        for point in points_3d:
            # Ray direction from camera through point
            ray_dir = point - cam_center
            
            # Normalize ray direction
            ray_dir_norm = np.linalg.norm(ray_dir)
            if ray_dir_norm < 1e-6:
                continue
            ray_dir = ray_dir / ray_dir_norm

            # Intersection with ground plane: z = ground_height
            # cam_center + t * ray_dir has z = ground_height
            # t = (ground_height - cam_center[2]) / ray_dir[2]
            
            if abs(ray_dir[2]) < 1e-6:
                # Ray is parallel to ground plane
                continue

            t = (ground_height - cam_center[2]) / ray_dir[2]
            
            if t < 0:
                # Point is behind camera
                continue

            # Intersection point
            intersection = cam_center + t * ray_dir
            projected_points.append(intersection[:2])  # Only x, y

        return np.array(projected_points) if projected_points else None

    def get_view_polygon(
        self, ground_height: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Get the projected frustum as a polygon on the ground plane.

        Args:
            ground_height: Height of ground plane (default: 0.0)

        Returns:
            Array of shape (N, 2) with vertices of projected frustum polygon in (x, y),
            or None if projection fails
        """
        # Get frustum corners in camera frame
        frustum_corners_cam = self.get_frustum_corners_camera_frame()

        # Transform to world frame
        frustum_corners_world = self.transform_points_to_world(frustum_corners_cam)

        # Project to ground plane
        projected_corners = self.project_to_ground_plane(
            frustum_corners_world, ground_height
        )

        if projected_corners is None or len(projected_corners) == 0:
            return None

        # Order points to form a valid polygon (convex hull)
        center = projected_corners.mean(axis=0)
        angles = np.arctan2(projected_corners[:, 1] - center[1],
                            projected_corners[:, 0] - center[0])
        ordered_indices = np.argsort(angles)
        ordered_polygon = projected_corners[ordered_indices]

        return ordered_polygon

    def get_camera_center_world(self) -> np.ndarray:
        """
        Get camera center position in world coordinates.

        Returns:
            Array of shape (3,) with camera center position [x, y, z]
        """
        return self.t_inv.copy()

    def get_camera_forward_direction(self) -> np.ndarray:
        """
        Get camera forward direction (negative z-axis) in world frame.

        Returns:
            Array of shape (3,) with normalized forward direction
        """
        # Camera's -z axis in camera frame is [0, 0, -1]
        # Transformed to world frame
        forward_cam = np.array([0.0, 0.0, -1.0])
        forward_world = self.R_inv @ forward_cam
        return forward_world / np.linalg.norm(forward_world)


def create_camera_matrix(
    position: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Create extrinsic matrix from camera position and rotation.

    Args:
        position: Camera position in world frame (3,)
        rotation_matrix: 3x3 rotation matrix (world to camera)

    Returns:
        4x4 extrinsic matrix
    """
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = position
    return extrinsic


def create_intrinsic_matrix(
    focal_length_x: float,
    focal_length_y: float,
    principal_point_x: float,
    principal_point_y: float,
) -> np.ndarray:
    """
    Create camera intrinsic matrix from focal lengths and principal point.

    Args:
        focal_length_x: Focal length in x direction (pixels)
        focal_length_y: Focal length in y direction (pixels)
        principal_point_x: Principal point x coordinate (pixels)
        principal_point_y: Principal point y coordinate (pixels)

    Returns:
        3x3 intrinsic matrix
    """
    K = np.array([
        [focal_length_x, 0, principal_point_x],
        [0, focal_length_y, principal_point_y],
        [0, 0, 1],
    ])
    return K
