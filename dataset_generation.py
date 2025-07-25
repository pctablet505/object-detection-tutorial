import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict

class ShapeDatasetGenerator:
    def _get_bbox_from_polygon(self, polygon: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box from polygon coordinates."""
        pts = polygon.reshape(-1, 2)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return int(x1), int(y1), int(x2), int(y2)

    def _get_bbox_from_circle(self, center: Tuple[int, int], radius: int) -> Tuple[int, int, int, int]:
        x1 = center[0] - radius
        y1 = center[1] - radius
        x2 = center[0] + radius
        y2 = center[1] + radius
        return int(x1), int(y1), int(x2), int(y2)

    def _get_bbox_from_ellipse(self, center: Tuple[int, int], axis1: int, axis2: int, angle: int) -> Tuple[int, int, int, int]:
        # More accurate bounding box for rotated ellipse
        angle_rad = math.radians(angle)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        
        # Calculate the actual bounding box considering rotation
        width = axis1 * cos_a + axis2 * sin_a
        height = axis1 * sin_a + axis2 * cos_a
        
        x1 = center[0] - width
        y1 = center[1] - height
        x2 = center[0] + width
        y2 = center[1] + height
        return int(x1), int(y1), int(x2), int(y2)

    def _is_within_bounds(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if a bounding box is within the image dimensions."""
        x1, y1, x2, y2 = bbox
        return x1 >= 0 and y1 >= 0 and x2 < self.img_size and y2 < self.img_size

    """
    A class to generate synthetic datasets of random geometric shapes.
    """
    
    def __init__(self, img_size: int = 256, margin: int = 40):
        self.img_size = img_size
        self.margin = margin
        self.shape_classes = ['circle', 'ellipse', 'triangle', 'rectangle', 'square', 'irregular']
    
    def _generate_random_color(self) -> Tuple[int, int, int]:
        """Generate a random BGR color."""
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    def _generate_random_center(self) -> Tuple[int, int]:
        """Generate a random center point within image boundaries."""
        return (
            random.randint(self.margin, self.img_size - self.margin),
            random.randint(self.margin, self.img_size - self.margin)
        )
    
    def _rotate_points(self, points: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Rotate points around the origin."""
        angle_rad = math.radians(angle_degrees)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ])
        
        points_2d = points.reshape(-1, 2)
        rotated = np.dot(points_2d, rotation_matrix.T)
        return rotated.reshape(-1, 1, 2).astype(np.int32)
    
    def _generate_circle(self) -> Tuple[str, callable]:
        """Generate parameters for a circle."""
        def draw_circle(image, color, center):
            radius = random.randint(self.img_size // 10, self.img_size // 4)
            cv2.circle(image, center, radius, color, -1)
            return radius * 2  # Return approximate size for overlap checking
        
        return 'circle', draw_circle
    
    def _generate_ellipse(self) -> Tuple[str, callable]:
        """Generate parameters for an ellipse."""
        def draw_ellipse(image, color, center):
            axis1 = random.randint(self.img_size // 10, self.img_size // 4)
            axis2 = random.randint(self.img_size // 10, self.img_size // 4)
            angle = random.randint(0, 360)
            cv2.ellipse(image, center, (axis1, axis2), angle, 0, 360, color, -1)
            return max(axis1, axis2) * 2  # Return approximate size
        
        return 'ellipse', draw_ellipse
    
    def _generate_triangle(self) -> Tuple[str, np.ndarray]:
        """Generate vertices for a triangle centered at origin."""
        max_coord = self.img_size // 4
        points = np.random.randint(-max_coord, max_coord, size=(3, 2))
        return 'triangle', points.reshape(-1, 1, 2)
    
    def _generate_rectangle(self) -> Tuple[str, np.ndarray]:
        """Generate vertices for a rectangle centered at origin."""
        max_size = self.img_size // 3
        w = random.randint(self.margin, max_size)
        h = random.randint(self.margin, max_size)
        half_w, half_h = w // 2, h // 2
        
        points = np.array([
            [-half_w, -half_h], [half_w, -half_h],
            [half_w, half_h], [-half_w, half_h]
        ], dtype=np.int32)
        
        return 'rectangle', points.reshape(-1, 1, 2)
    
    def _generate_square(self) -> Tuple[str, np.ndarray]:
        """Generate vertices for a square centered at origin."""
        max_size = self.img_size // 3
        side = random.randint(self.margin, max_size)
        half_side = side // 2
        
        points = np.array([
            [-half_side, -half_side], [half_side, -half_side],
            [half_side, half_side], [-half_side, half_side]
        ], dtype=np.int32)
        
        return 'square', points.reshape(-1, 1, 2)
    
    def _generate_irregular_polygon(self) -> Tuple[str, np.ndarray]:
        """Generate vertices for an irregular polygon centered at origin."""
        num_vertices = random.randint(4, 8)
        max_radius = self.img_size // 4
        
        vertices = []
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
        
        for angle in angles:
            radius = random.uniform(self.img_size // 10, max_radius)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append((int(x), int(y)))
        
        points = np.array(vertices, dtype=np.int32).reshape(-1, 1, 2)
        return f'irregular_{num_vertices}', points
    
    def _generate_shape(self, shape_class: str) -> Tuple[str, object]:
        """Generate a shape based on the class name."""
        shape_generators = {
            'circle': self._generate_circle,
            'ellipse': self._generate_ellipse,
            'triangle': self._generate_triangle,
            'rectangle': self._generate_rectangle,
            'square': self._generate_square,
            'irregular': self._generate_irregular_polygon
        }
        
        return shape_generators[shape_class]()
    
    def _check_overlap(self, new_center: Tuple[int, int], new_size: int, 
                      existing_objects: List[Dict], min_distance: int = 20) -> bool:
        """Check if a new object would overlap with existing objects."""
        for obj in existing_objects:
            existing_center = obj['center']
            existing_size = obj.get('size', 50)  # Default size if not available
            
            # Calculate distance between centers
            distance = math.sqrt(
                (new_center[0] - existing_center[0])**2 + 
                (new_center[1] - existing_center[1])**2
            )
            
            # Check if objects would overlap (with some minimum distance)
            required_distance = (new_size + existing_size) / 2 + min_distance
            if distance < required_distance:
                return True
        
        return False
    
    def _is_within_bounds(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if a bounding box is within the image dimensions."""
        x1, y1, x2, y2 = bbox
        return x1 >= 0 and y1 >= 0 and x2 < self.img_size and y2 < self.img_size
    
    def generate_single_image(self, max_objects: int = 1, 
                            allow_empty: bool = True,
                            max_attempts: int = 50) -> Dict:
        """
        Generate a single image with multiple random shapes.
        
        Args:
            max_objects: Maximum number of objects to place in the image
            allow_empty: Whether to allow images with 0 objects
            max_attempts: Maximum attempts to place each object without overlap
        """
        # Create blank canvas
        image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Determine number of objects to place
        min_objects = 0 if allow_empty else 1
        num_objects = random.randint(min_objects, max_objects)
        
        objects = []
        placed_objects = []
        for obj_id in range(num_objects):
            placed = False
            attempts = 0
            while not placed and attempts < max_attempts:
                shape_class = random.choice(self.shape_classes)
                final_class_name, shape_data = self._generate_shape(shape_class)
                color = self._generate_random_color()
                center = self._generate_random_center()
                estimated_size = self.img_size // 4  # Increased size estimate
                if not self._check_overlap(center, estimated_size, placed_objects):
                    # Generate the shape first to get accurate bbox
                    polygon = None
                    bbox = None
                    actual_size = estimated_size
                    
                    if callable(shape_data):
                        if shape_class == 'circle':
                            radius = random.randint(self.img_size // 10, self.img_size // 4)
                            # Check if circle fits within bounds
                            bbox = self._get_bbox_from_circle(center, radius)
                            if self._is_within_bounds(bbox):
                                cv2.circle(image, center, radius, color, -1)
                                polygon = np.array([
                                    [center[0] + radius, center[1]],
                                    [center[0], center[1] + radius],
                                    [center[0] - radius, center[1]],
                                    [center[0], center[1] - radius]
                                ], dtype=np.int32).reshape(-1, 1, 2)
                                actual_size = radius * 2
                            else:
                                continue  # Try again with new position
                        elif shape_class == 'ellipse':
                            axis1 = random.randint(self.img_size // 10, self.img_size // 4)
                            axis2 = random.randint(self.img_size // 10, self.img_size // 4)
                            angle = random.randint(0, 360)
                            # Check if ellipse fits within bounds
                            bbox = self._get_bbox_from_ellipse(center, axis1, axis2, angle)
                            if self._is_within_bounds(bbox):
                                cv2.ellipse(image, center, (axis1, axis2), angle, 0, 360, color, -1)
                                polygon = np.array([
                                    [center[0] + axis1, center[1]],
                                    [center[0], center[1] + axis2],
                                    [center[0] - axis1, center[1]],
                                    [center[0], center[1] - axis2]
                                ], dtype=np.int32).reshape(-1, 1, 2)
                                actual_size = max(axis1, axis2) * 2
                            else:
                                continue  # Try again with new position
                    else:
                        rotation_angle = random.uniform(0, 360)
                        rotated_vertices = self._rotate_points(shape_data, rotation_angle)
                        translation = np.array([center[0], center[1]]).reshape(1, 1, 2)
                        final_vertices = rotated_vertices + translation
                        polygon = final_vertices.astype(np.int32)
                        bbox = self._get_bbox_from_polygon(polygon)
                        
                        # Check if polygon fits within bounds
                        if self._is_within_bounds(bbox):
                            cv2.fillPoly(image, [final_vertices.astype(np.int32)], color)
                            vertices_2d = final_vertices.reshape(-1, 2)
                            min_x, max_x = vertices_2d[:, 0].min(), vertices_2d[:, 0].max()
                            min_y, max_y = vertices_2d[:, 1].min(), vertices_2d[:, 1].max()
                            actual_size = max(max_x - min_x, max_y - min_y)
                        else:
                            continue  # Try again with new position
                    
                    # Only add object if it was successfully placed within bounds
                    if bbox is not None and polygon is not None:
                        obj_info = {
                            'class': final_class_name,
                            'polygon': polygon.reshape(-1, 2).tolist(),
                            'bbox': bbox
                        }
                        objects.append(obj_info)
                        placed_objects.append({'center': center, 'size': actual_size})
                        placed = True
                attempts += 1
            if not placed:
                print(f"Warning: Could not place object {obj_id} after {max_attempts} attempts")
        return {
            'image': image,
            'objects': objects,
            'num_objects': len(objects)
        }
    
    def generate_dataset(self, num_images: int, max_objects_per_image: int = 3, 
                        allow_empty_images: bool = True) -> List[Dict]:
        """
        Generate a complete dataset of images.
        
        Args:
            num_images: Number of images to generate
            max_objects_per_image: Maximum number of objects per image
            allow_empty_images: Whether to allow images with 0 objects
        """
        dataset = []
        
        print(f"Generating {num_images} images with up to {max_objects_per_image} objects each...")
        
        for i in range(num_images):
            sample = self.generate_single_image(
                max_objects=max_objects_per_image,
                allow_empty=allow_empty_images
            )
            sample['image_id'] = i
            dataset.append(sample)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_images} images")
        
        print(f"Successfully generated {len(dataset)} images.")
        return dataset
    
    def get_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """Get statistics about the generated dataset."""
        stats = {
            'total_images': len(dataset),
            'empty_images': 0,
            'total_objects': 0,
            'objects_per_image': [],
            'class_counts': {},
            'avg_objects_per_image': 0
        }
        
        for sample in dataset:
            num_objects = sample['num_objects']
            stats['objects_per_image'].append(num_objects)
            stats['total_objects'] += num_objects
            
            if num_objects == 0:
                stats['empty_images'] += 1
            
            # Count classes
            for obj in sample['objects']:
                class_name = obj['class']
                stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
        
        if len(dataset) > 0:
            stats['avg_objects_per_image'] = stats['total_objects'] / len(dataset)
        
        return stats
    
    def visualize_samples(self, dataset: List[Dict], num_samples: int = 20):
        """Visualize samples from the dataset."""
        num_samples = min(num_samples, len(dataset))
        rows = int(np.ceil(num_samples / 5))
        
        fig, axes = plt.subplots(rows, 5, figsize=(20, 4 * rows))
        
        # Handle case where we have only one row
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            sample = dataset[i]
            row, col = i // 5, i % 5
            
            # Convert BGR to RGB for matplotlib
            rgb_image = cv2.cvtColor(sample['image'], cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(rgb_image)
            
            # Create title with object info
            title = f"ID: {sample['image_id']} | Objects: {sample['num_objects']}"
            if sample['num_objects'] > 0:
                classes = [obj['class'] for obj in sample['objects']]
                title += f"\nClasses: {', '.join(classes[:3])}"  # Show first 3 classes
                if len(classes) > 3:
                    title += "..."
            
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * 5):
            row, col = i // 5, i % 5
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function."""
    # Configuration
    IMG_SIZE = 256
    NUM_IMAGES = 50
    MAX_OBJECTS_PER_IMAGE = 1  # New parameter
    ALLOW_EMPTY_IMAGES = True  # New parameter
    NUM_SAMPLES_TO_SHOW = 20
    
    # Create generator
    generator = ShapeDatasetGenerator(img_size=IMG_SIZE, margin=30)
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_images=NUM_IMAGES,
        max_objects_per_image=MAX_OBJECTS_PER_IMAGE,
        allow_empty_images=ALLOW_EMPTY_IMAGES
    )
    
    # Print dataset statistics
    stats = generator.get_dataset_statistics(dataset)
    print(f"\n--- Dataset Statistics ---")
    print(f"Total images: {stats['total_images']}")
    print(f"Empty images: {stats['empty_images']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Average objects per image: {stats['avg_objects_per_image']:.2f}")
    print(f"Objects per image distribution: min={min(stats['objects_per_image'])}, "
          f"max={max(stats['objects_per_image'])}")
    
    print(f"\n--- Class Distribution ---")
    for class_name, count in sorted(stats['class_counts'].items()):
        print(f"{class_name}: {count} objects")
    
    # Visualize samples
    print(f"\n--- Visualizing {NUM_SAMPLES_TO_SHOW} samples ---")
    generator.visualize_samples(dataset, NUM_SAMPLES_TO_SHOW)
    
    return dataset


if __name__ == '__main__':
    dataset = main()