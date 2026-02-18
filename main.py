#!/usr/bin/env python3
"""
NerdScan - Photo Detection and Extraction Tool

This script detects and extracts photos from scanned images using Grounded SAM.
It creates both cropped photos and visualizations of the detection process.
"""

import argparse
import os
import cv2
import numpy as np
import torch
import logging
import random
import re
import datetime
import subprocess
from pathlib import Path
from PIL import Image
import piexif
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForVision2Seq
from collections import defaultdict
from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler
from rich import print as rprint

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.grounding_dino.processing_grounding_dino")
warnings.filterwarnings("ignore", message=".*Image size.*exceeds limit.*could be decompression bomb DOS attack.*")

# Disable tokenizer parallelism to avoid warnings in multiprocessing contexts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up rich console
console = Console()

# Set up logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger('NerdScan')

# Emoji constants for better UI
EMOJI_PHOTO = "📷"
EMOJI_SUCCESS = "✅"
EMOJI_WARNING = "⚠️"
EMOJI_ERROR = "❌"
EMOJI_ROCKET = "🚀"
EMOJI_FOLDER = "📁"
EMOJI_CLOCK = "🕒"
EMOJI_MAGIC = "✨"

class PhotoDetector:
    """
    Photo detector that uses Grounded SAM to find and extract photos from scanned images.
    """
    
    def __init__(self, model_id="IDEA-Research/grounding-dino-base", device=None, seed=42):
        """
        Initialize the detector with the specified model.
        
        Args:
            model_id (str): The Hugging Face model ID for object detection
            device (str): Device to run the model on ('cuda' or 'cpu')
            seed (int): Random seed for reproducibility
        """
        # Set random seed for reproducibility
        if seed is not None:
            logger.info(f"{EMOJI_MAGIC} Setting random seed to {seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"{EMOJI_ROCKET} Using device: [bold cyan]{self.device}[/bold cyan]")
        
        # Load text-based object detection model
        with console.status(f"[bold green]Loading model: {model_id}...[/bold green]", spinner="dots"):
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        logger.info(f"{EMOJI_SUCCESS} Model loaded successfully")

    def add_border(self, image, border_size=100):
        new_width = image.width + 2 * border_size
        new_height = image.height + 2 * border_size
        bordered_image = Image.new('RGB', (new_width, new_height), 'white')
        bordered_image.paste(image, (border_size, border_size))
        logger.info(f"Added {border_size}px white border. New size: {image.size[0]}x{image.size[1]}")
        return bordered_image
    
    def detect_photos(self, image, text_prompt="a photo. a picture.",
                     box_threshold=0.05, text_threshold=0.05, confidence_threshold=0.15):
        """
        Detect photos in a scanned image using text prompts.
        
        Args:
            image : PIL image data
            text_prompt (str): Text prompt for detection (lowercase, end with period)
            box_threshold (float): Confidence threshold for bounding boxes
            text_threshold (float): Confidence threshold for text
            confidence_threshold (float): Final confidence threshold for keeping detections
            add_border (bool): Add 20px white border around the image before detection
            
        Returns:
            tuple: (original_image, bounding_boxes, scores, labels)
        """
        
        # Ensure text prompt is properly formatted
        if not text_prompt.islower() or not text_prompt.endswith('.'):
            logger.warning(f"{EMOJI_WARNING} Text prompt should be lowercase and end with a period.")
            text_prompt = text_prompt.lower()
            if not text_prompt.endswith('.'):
                text_prompt = text_prompt + '.'
        
        # Process image with the model
        with console.status(f"[bold green]Running inference...[/bold green]", spinner="dots"):
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]]  # [H, W] format
            )
        
        if not results or len(results) == 0 or "boxes" not in results[0]:
            logger.warning(f"{EMOJI_WARNING} No photos detected")
            return image, [], [], []
        
        # Extract results
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"]
        
        # Apply confidence threshold filtering
        high_confidence_indices = np.where(scores >= confidence_threshold)[0]
        if len(high_confidence_indices) == 0:
            logger.warning(f"{EMOJI_WARNING} No photos with confidence >= {confidence_threshold} found.")
            return image, [], [], []
            
        boxes = boxes[high_confidence_indices]
        scores = scores[high_confidence_indices]
        labels = [labels[i] for i in high_confidence_indices]
        
        logger.info(f"{EMOJI_SUCCESS} Found {len(boxes)} potential photos")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            logger.info(f"  Photo {i+1}: {label} (Score: [bold green]{score:.4f}[/bold green]), Box: {box}")
        
        return image, boxes, scores, labels
    
    def calculate_overlap_percentage(self, box1, box2):
        """
        Calculate the overlap percentage between two bounding boxes.
        
        Args:
            box1 (list): First bounding box in [x1, y1, x2, y2] format
            box2 (list): Second bounding box in [x1, y1, x2, y2] format
            
        Returns:
            float: Percentage of the smallest box area that overlaps
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection coordinates
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # Check if boxes overlap
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate overlap percentage based on the smaller area
        min_area = min(area1, area2)
        overlap_percentage = (intersection_area / min_area) * 100.0
        
        return overlap_percentage
    
    def remove_overlapping_boxes(self, boxes, scores, labels, overlap_threshold=0.05):
        """
        Remove overlapping bounding boxes, keeping only the ones with higher confidence scores.
        
        Args:
            boxes (list): List of bounding boxes in [x1, y1, x2, y2] format
            scores (list): Confidence scores for each detection
            labels (list): Labels for each detection
            overlap_threshold (float): Ratio threshold for considering boxes as overlapping (0.0 to 1.0)
            
        Returns:
            tuple: (filtered_boxes, filtered_scores, filtered_labels)
        """
        if len(boxes) <= 1:
            return boxes, scores, labels
        
        logger.info(f"Checking for overlapping boxes with threshold {overlap_threshold:.2f} (ratio)")
        
        # Convert to list for easier manipulation
        boxes_list = boxes.tolist() if isinstance(boxes, np.ndarray) else list(boxes)
        scores_list = scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
        labels_list = list(labels)
        
        # Sort indices by confidence score (highest first)
        indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
        
        # Initialize list to keep track of which boxes to keep
        keep = [True] * len(boxes_list)
        
        # Check each box against others with lower confidence
        for i, idx1 in enumerate(indices):
            if not keep[idx1]:
                continue  # Skip if already marked for removal
            
            box1 = boxes_list[idx1]
            
            # Compare with remaining boxes of lower confidence
            for idx2 in indices[i+1:]:
                if not keep[idx2]:
                    continue  # Skip if already marked for removal
                
                box2 = boxes_list[idx2]
                
                # Calculate overlap percentage
                overlap = self.calculate_overlap_percentage(box1, box2)
                
                # If overlap exceeds threshold, mark the lower confidence box for removal
                if overlap > overlap_threshold * 100:  # Convert ratio to percentage for comparison
                    logger.info(f"  Removing box with score {scores_list[idx2]:.4f} due to {overlap:.2f}% overlap")
                    keep[idx2] = False
        
        # Filter boxes, scores, and labels
        filtered_boxes = [boxes_list[i] for i in range(len(boxes_list)) if keep[i]]
        filtered_scores = [scores_list[i] for i in range(len(scores_list)) if keep[i]]
        filtered_labels = [labels_list[i] for i in range(len(labels_list)) if keep[i]]
        
        logger.info(f"{EMOJI_SUCCESS} Removed {len(boxes) - len(filtered_boxes)} overlapping boxes, kept {len(filtered_boxes)}")
        
        # Convert back to numpy arrays if original were numpy arrays
        if isinstance(boxes, np.ndarray):
            filtered_boxes = np.array(filtered_boxes)
        if isinstance(scores, np.ndarray):
            filtered_scores = np.array(filtered_scores)
        
        return filtered_boxes, filtered_scores, filtered_labels

    def remove_large_small_boxes(self, boxes, scores, labels, size, min_percent=0., max_percent=1.):
        remain = []
        for i, box in enumerate(boxes):
            # [x1, y1, x2, y2]
            cur_size = abs((box[2] - box[0]) * (box[3] - box[1]))
            if min_percent <= cur_size/size <= max_percent:
                remain.append(True)
            else:
                logger.info(f' Removing box {box} with size {cur_size/size:.2f}')
                remain.append(False)
        boxes = [b for i, b in enumerate(boxes) if remain[i]]
        scores = [s for i, s in enumerate(scores) if remain[i]]
        labels = [l for i, l in enumerate(labels) if remain[i]]
        return boxes, scores, labels

    
    def create_visualization(self, image, boxes, scores, labels, output_paths, output_vis_path):
        """
        Create visualization showing original image with bounding boxes and cropped results.
        
        Args:
            image: image data
            boxes (list): List of bounding boxes
            scores (list): Confidence scores for each detection
            labels (list): Labels for each detection
            output_paths (list): Paths to saved cropped images
            output_vis_path (str): Path to save the visualization
        """
        
        # Load the original image with PIL to preserve colors
        original_pil = image
        
        # Create figure
        if output_paths and len(output_paths) > 0:
            # Calculate figure size based on number of crops
            fig_width = min(20, 5 + 4 * len(output_paths))
            fig, axs = plt.subplots(1, 1 + len(output_paths), figsize=(fig_width, 8))
            
            # Display original image with bounding boxes
            axs[0].imshow(original_pil)
            axs[0].set_title('Original with Detections')
            
            # Draw bounding boxes
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                axs[0].add_patch(rect)
                axs[0].text(x1, y1-10, f"{label}: {score:.2f}", color='red', fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.7))
            
            # Remove axis ticks
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            
            # Display cropped images
            for i, (output_path, score, label) in enumerate(zip(output_paths, scores, labels)):
                cropped_img = Image.open(output_path)
                axs[i+1].imshow(cropped_img)
                axs[i+1].set_title(f'Crop {i+1}: {label} ({score:.2f})')
                axs[i+1].set_xticks([])
                axs[i+1].set_yticks([])
        else:
            # Just show original with bounding boxes
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(original_pil)
            ax.set_title('Detections')
            
            # Draw bounding boxes
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-10, f"{label}: {score:.2f}", color='red', fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_vis_path, dpi=150)
        plt.close()
        logger.info(f"{EMOJI_SUCCESS} Visualization saved to [cyan]{output_vis_path}[/cyan]")


    def estimate_orientation(self, pil_image):
        """
        Estimate image orientation in multiples of 90 degrees using SmolVLM-256M-Instruct.
        Returns one of {1, 3, 6, 8} corresponding to {0°, 180°, 90° CW, 270° CW}.
        """
        try:
            # Load SmolVLM model and processor (cached for efficiency)
            if not hasattr(self, '_smolvlm_model'):
                with console.status("[bold green]Loading SmolVLM model for orientation detection...[/bold green]", spinner="dots"):
                    self._smolvlm_model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
                    self._smolvlm_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
                    self._smolvlm_model.to(self.device)
                logger.info(f"{EMOJI_SUCCESS} SmolVLM model loaded for orientation detection")

            # Define the messages for orientation estimation (SmolVLM uses chat template)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What is the orientation of this image? Answer with one of: normal, rotated 90 degrees clockwise, rotated 180 degrees, rotated 270 degrees clockwise."}
                    ]
                }
            ]

            # Apply chat template
            prompt = self._smolvlm_processor.apply_chat_template(messages, add_generation_prompt=True)

            # Process image and text
            inputs = self._smolvlm_processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self._smolvlm_model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = self._smolvlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Parse the response to determine orientation
            response_lower: str = generated_text.lower().strip()
            response_lower = response_lower[response_lower.rfind('assistant:'):]
            logger.info(f'model response: {response_lower}')
            orientation_map = {
                "normal": 1,
                "rotated 90 degrees clockwise": 6,
                "rotated 180 degrees": 3,
                "rotated 270 degrees clockwise": 8
            }

            best_orientation = None
            for key, value in orientation_map.items():
                if key in response_lower:
                    best_orientation = value
                    break

            if best_orientation is None:
                logger.warning(f"{EMOJI_WARNING} Could not parse orientation from response: {generated_text}, defaulting to Normal")
                return 1

            logger.info(f"{EMOJI_MAGIC} Estimated orientation: {best_orientation} (response: {generated_text.strip()})")
            return best_orientation

        except Exception as e:
            logger.warning(f"{EMOJI_WARNING} SmolVLM orientation estimation failed: {e}, defaulting to Normal")
            return 1

def process_images(detector, input_dir, output_dir, vis_dir, text_prompt,
                  preserve_structure=False, remove_overlaps=False, 
                  overlap_threshold=0.05, confidence_threshold=0.15,
                  sample_size=None, seed=42, auto_orient=False, rotate_pixels=False, add_border=True):
    """
    Process all images in the input directory and save results.
    
    Args:
        detector: Initialized PhotoDetector instance
        input_dir (str): Input directory containing scanned images
        output_dir (str): Output directory for cropped photos
        vis_dir (str): Output directory for visualizations
        text_prompt (str): Text prompt for detection
        preserve_structure (bool): Whether to preserve folder structure
        remove_overlaps (bool): Whether to remove overlapping boxes
        overlap_threshold (float): Threshold for overlap detection
        confidence_threshold (float): Minimum confidence score to keep detections
        sample_size (int): Number of random images to process (None = all)
        seed (int): Random seed for reproducibility
        auto_orient (bool): Estimate orientation of each cropped photo and write EXIF Orientation.
        rotate_pixels (bool): If True, physically rotate pixels to upright and set Orientation=1.
        add_border (bool): Add 20px white border around images before detection. Default: True.
    """
    console.print(f"\n{EMOJI_ROCKET} [bold green]Starting NerdScan processing[/bold green]")
    console.print(f"{EMOJI_FOLDER} Input: [cyan]{input_dir}[/cyan]")
    console.print(f"{EMOJI_FOLDER} Output crops: [cyan]{output_dir}[/cyan]")
    console.print(f"{EMOJI_FOLDER} Output visualizations: [cyan]{vis_dir}[/cyan]")
    console.print(f"{EMOJI_MAGIC} Text prompt: [yellow]\"{text_prompt}\"[/yellow]")
    console.print(f"{EMOJI_MAGIC} Confidence threshold: [yellow]{confidence_threshold}[/yellow]")
    if remove_overlaps:
        console.print(f"{EMOJI_MAGIC} Removing overlaps with threshold: [yellow]{overlap_threshold:.2f}[/yellow]")
    if preserve_structure:
        console.print(f"{EMOJI_MAGIC} Preserving folder structure")
    if auto_orient:
        console.print(f"{EMOJI_MAGIC} Auto-orientation enabled (rotate_pixels={'ON' if rotate_pixels else 'OFF'})")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Track all image files
    all_image_files = []
    with console.status("[bold green]Scanning for image files...[/bold green]", spinner="dots"):
        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')):
                    l, ext = os.path.splitext(filename)
                    if l.endswith('_alt'):  # 修改的直接跳过，实际会被原始图片调用到。
                        continue
                    full_path = os.path.join(root, filename)
                    all_image_files.append(full_path)

    # Sort the list of files alphabetically by path
    all_image_files.sort()

    console.print(f"{EMOJI_PHOTO} Found [bold]{len(all_image_files)}[/bold] image files")
    
    # Sample random images if requested
    if sample_size and sample_size < len(all_image_files):
        random.seed(seed)
        selected_files = random.sample(all_image_files, sample_size)
        console.print(f"{EMOJI_MAGIC} Selected [bold]{sample_size}[/bold] random images for processing")
    else:
        selected_files = all_image_files
        console.print(f"{EMOJI_PHOTO} Processing all [bold]{len(selected_files)}[/bold] images")
    
    # Track statistics
    processed_files = 0
    found_photos = 0
    year_photo_count = defaultdict(int)
    
    # Process each selected file
    for input_path in tqdm(selected_files, desc="Processing images", unit="image"):
        # Check parent directory name for year
        root = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        parent_dir_name = os.path.basename(root)
        year = None

        l, ext = os.path.splitext(input_path)
        if os.path.isfile(f'{l}_alt.{ext}'):
            input_path = f'{l}_alt.{ext}'  # 如果手动修改过，则用这个作为输入图片地址。

        # load image
        logger.info(f"{EMOJI_PHOTO} Processing image: [cyan]{input_path}[/cyan]")

        # Load image
        try:
            image = Image.open(input_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Increase the pixel limit for Pillow
            Image.MAX_IMAGE_PIXELS = None  # Disable the limit
            logger.info(f"Image loaded: {image.size[0]}x{image.size[1]}")
        except Exception as e:
            logger.error(f"{EMOJI_ERROR} Error loading image {input_path}: {e}")
            return None, [], [], []

        if add_border:
            image = detector.add_border(image, border_size=30)
        
        # Check if parent_dir_name is a 4-digit year
        if re.match(r'^\d{4}$', parent_dir_name):
            try:
                year_num = int(parent_dir_name)
                current_year = datetime.datetime.now().year
                # Basic sanity check for plausible years
                if 1800 < year_num <= current_year:
                    year = year_num
                    logger.info(f"{EMOJI_CLOCK} Detected year {year} from folder '{parent_dir_name}' for EXIF")
            except ValueError:
                pass  # Not a valid integer
        
        try:
            # Detect photos in the image with confidence threshold
            image, boxes, scores, labels = detector.detect_photos(
                image, text_prompt, confidence_threshold=confidence_threshold
            )
            
            if image is None or len(boxes) == 0:
                continue
            
            # Remove overlapping boxes if requested
            if remove_overlaps and len(boxes) > 1:
                # remove boxes > 80%
                boxes, scores, labels = detector.remove_large_small_boxes(
                    boxes, scores, labels,size=image.size[0]*image.size[1],
                    min_percent=0.1,
                    max_percent=0.8)

                boxes, scores, labels = detector.remove_overlapping_boxes(
                    boxes, scores, labels, overlap_threshold
                )
                if len(boxes) == 0:
                    logger.warning(f"{EMOJI_WARNING} All boxes were removed due to overlap in {input_path}")
                    continue
            
            # Determine output directories based on preserve_structure flag
            if preserve_structure:
                # Create corresponding output subdirectory matching input structure
                relative_path = os.path.relpath(root, input_dir)
                curr_output_dir = os.path.join(output_dir, relative_path)
                curr_vis_dir = os.path.join(vis_dir, relative_path)
            else:
                # Flat structure - all outputs go directly in output dirs
                curr_output_dir = output_dir
                curr_vis_dir = vis_dir
            
            # Create output directories if they don't exist
            os.makedirs(curr_output_dir, exist_ok=True)
            os.makedirs(curr_vis_dir, exist_ok=True)
            
            # Base filename without extension
            base_filename = os.path.splitext(filename)[0]
            output_paths = []
            
            # Process each detected photo
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                # Convert to integers and ensure box is within image boundaries
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
                
                # Skip if box is too small
                min_size = 100  # Minimum size in pixels
                if (x2 - x1 < min_size) or (y2 - y1 < min_size):
                    logger.warning(f"{EMOJI_WARNING} Skipping box {i+1} - too small: {x2-x1}x{y2-y1}")
                    continue
                
                # Crop the image
                cropped_pil = image.crop((x1, y1, x2, y2))

                # Auto orientation estimation and optional rotation
                exif_orientation = None
                if auto_orient:
                    exif_orientation = detector.estimate_orientation(cropped_pil)
                    if rotate_pixels and exif_orientation in (3, 6, 8):
                        # Apply rotation to pixels and reset orientation to Normal
                        if exif_orientation == 3:
                            cropped_pil = cropped_pil.rotate(180, expand=True)
                        elif exif_orientation == 6:
                            cropped_pil = cropped_pil.rotate(-90, expand=True)  # 90 CW
                        elif exif_orientation == 8:
                            cropped_pil = cropped_pil.rotate(90, expand=True)   # 270 CW
                        exif_orientation = 1
                
                # Generate output filename
                if year:
                    # Update the count for this year
                    year_photo_count[year] += 1
                    photo_index = year_photo_count[year]
                    
                    # Create a sequential date within the year
                    day_of_year = min(photo_index, 365)  # Cap at 365 days
                    date_in_year = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year-1)
                    month = date_in_year.month
                    day = date_in_year.day
                    
                    # Format with leading zeros for EXIF
                    exif_date_str = f"{year}:{month:02d}:{day:02d} 12:00:00"
                    
                    # Filename with year and sequential index
                    output_filename = f"{base_filename}_{year}_{photo_index:03d}.jpg"
                else:
                    exif_date_str = None
                    output_filename = f"{base_filename}_{i+1:03d}.jpg"
                
                # Save path
                output_path = os.path.join(curr_output_dir, output_filename)
                
                # Save the cropped image
                cropped_pil.save(output_path, quality=95)
                output_paths.append(output_path)

                # Write EXIF metadata: orientation (if detected) and optional dates
                try:
                    zeroth_ifd = {
                        piexif.ImageIFD.Make: "NerdScan",
                        piexif.ImageIFD.Software: "PhotoDetector"
                    }
                    if exif_orientation is not None:
                        zeroth_ifd[piexif.ImageIFD.Orientation] = int(exif_orientation)

                    exif_ifd = {}
                    if exif_date_str:
                        exif_ifd.update({
                            piexif.ExifIFD.DateTimeOriginal: exif_date_str,
                            piexif.ExifIFD.DateTimeDigitized: exif_date_str
                        })

                    exif_dict = {"0th": zeroth_ifd}
                    if exif_ifd:
                        exif_dict["Exif"] = exif_ifd
                    exif_bytes = piexif.dump(exif_dict)

                    # Re-save with EXIF
                    img = Image.open(output_path)
                    img.save(output_path, exif=exif_bytes, quality=95)

                    # exiftool enhancement if available
                    try:
                        cmd = ['exiftool', '-overwrite_original']
                        if exif_date_str:
                            # Parse parts for IPTC format
                            date_parts = exif_date_str.split()[0].split(':')
                            iptc_date_str = f"{date_parts[0]}{date_parts[1]}{date_parts[2]}"
                            cmd.extend([
                                f'-EXIF:DateTimeOriginal={exif_date_str}',
                                f'-EXIF:CreateDate={exif_date_str}',
                                f'-EXIF:ModifyDate={exif_date_str}',
                                f'-XMP:DateCreated={exif_date_str}',
                                f'-XMP:CreateDate={exif_date_str}',
                                f'-XMP:ModifyDate={exif_date_str}',
                                f'-IPTC:DateCreated={iptc_date_str}',
                                f'-FileCreateDate={exif_date_str}',
                                f'-FileModifyDate={exif_date_str}',
                                f'-AllDates={exif_date_str}',
                            ])
                        if exif_orientation is not None:
                            cmd.append(f'-Orientation#={int(exif_orientation)}')
                        cmd.extend(['-E', '-F', output_path])
                        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                        if result.returncode == 0:
                            logger.info(f"{EMOJI_SUCCESS} Updated metadata via exiftool for {output_path}")
                        else:
                            logger.warning(f"{EMOJI_WARNING} Error running exiftool: {result.stderr}")
                    except Exception as e:
                        logger.warning(f"{EMOJI_WARNING} Failed to run exiftool (not installed?): {e}")

                    if exif_date_str:
                        logger.info(f"{EMOJI_CLOCK} Set EXIF date {exif_date_str} for {output_path}")
                    if exif_orientation is not None:
                        logger.info(f"{EMOJI_CLOCK} Set EXIF orientation {int(exif_orientation)} for {output_path}")
                except Exception as e:
                    logger.error(f"{EMOJI_ERROR} Error setting EXIF metadata: {e}")
                
                found_photos += 1
            
            # Only create visualization if we found and saved photos
            if output_paths:
                vis_path = os.path.join(curr_vis_dir, f"{base_filename}_visualization.jpg")
                detector.create_visualization(
                    image, boxes, scores, labels, output_paths, vis_path
                )
            
            processed_files += 1
            
        except Exception as e:
            logger.error(f"{EMOJI_ERROR} Error processing {input_path}: {e}")
    
    # Print summary
    console.print(f"\n{EMOJI_SUCCESS} [bold green]Processing complete![/bold green]")
    console.print(f"✓ Processed {processed_files} images")
    console.print(f"✓ Found {found_photos} photos")
    
    # Print count of photos per year
    if year_photo_count:
        console.print(f"\n{EMOJI_CLOCK} [bold]Photos extracted by year:[/bold]")
        for year, count in sorted(year_photo_count.items()):
            console.print(f"  {year}: [bold cyan]{count}[/bold cyan] photos")
    
    return processed_files, found_photos


def main():
    """Parse arguments and process images."""
    parser = argparse.ArgumentParser(
        description=f"{EMOJI_PHOTO} NerdScan - Detect and extract photos from scanned images."
    )
    parser.add_argument(
        "-i", "--input", default="input", 
        help="Input directory containing scanned images."
    )
    parser.add_argument(
        "-o", "--output", default="output", 
        help="Output directory to save extracted photos."
    )
    parser.add_argument(
        "--text-prompt", default="an old photo.",
        help="Text prompt for detection. Should be lowercase and end with a period."
    )
    parser.add_argument(
        "--single-image", 
        help="Process a single image instead of a directory."
    )
    parser.add_argument(
        "--preserve-structure", action="store_true", 
        help="Preserve folder structure from input to output. Default is flat structure."
    )
    parser.add_argument(
        "--remove-overlaps", action="store_true", default=True,
        help="Remove overlapping detection boxes, keeping those with higher confidence."
    )
    parser.add_argument(
        "--overlap-threshold", type=float, default=0.05,
        help="Ratio threshold for considering boxes as overlapping (0.0 to 1.0). A higher value allows more overlap. Default: 0.05."
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.15,
        help="Minimum confidence score to keep detections (0.0 to 1.0). Lower values find more potential photos but may increase false positives. Higher values are stricter. Default: 0.15."
    )
    parser.add_argument(
        "--auto-orient", action="store_true", default=False,
        help="Estimate orientation of each cropped photo and set EXIF Orientation (no pixel rotation)."
    )
    parser.add_argument(
        "--rotate-pixels", action="store_true", default=False,
        help="Physically rotate pixels upright when auto-orienting; Orientation tag will be set to Normal."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of random images to process. If not specified, process all images."
    )
    parser.add_argument(
        "--device", default=None, 
        help="Device to run the model on ('cuda' or 'cpu'). Default: use CUDA if available."
    )
    parser.add_argument(
        "--add-border", action="store_true", default=True,
        help="Add 20px white border around images before detection. Default: enabled."
    )
    parser.add_argument(
        "--no-border", action="store_true", default=False,
        help="Disable adding white border around images before detection."
    )
    
    args = parser.parse_args()
    
    # Handle border flag logic
    if args.no_border:
        args.add_border = False
    
    # Print welcome message
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]{' '*18}NerdScan {EMOJI_PHOTO} {EMOJI_MAGIC}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    # Ensure paths are absolute relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize detector
    detector = PhotoDetector(device=args.device, seed=args.seed)
    
    if args.single_image:
        # Process a single image
        if not os.path.isfile(args.single_image):
            console.print(f"{EMOJI_ERROR} [bold red]Error: Input file '{args.single_image}' not found.[/bold red]")
            return
        
        # Setup output directories
        output_dir = os.path.join(script_dir, args.output, "crops")
        vis_dir = os.path.join(script_dir, args.output, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get file name and potential year from parent directory
        input_path = os.path.abspath(args.single_image)
        parent_dir = os.path.basename(os.path.dirname(input_path))
        filename = os.path.basename(input_path)
        
        console.print(f"{EMOJI_PHOTO} Processing single image: [cyan]{input_path}[/cyan]")
        
        # Process this single image
        process_images(
            detector=detector,
            input_dir=os.path.dirname(input_path),
            output_dir=output_dir,
            vis_dir=vis_dir,
            text_prompt=args.text_prompt,
            preserve_structure=False,  # Not relevant for single image
            remove_overlaps=args.remove_overlaps,
            overlap_threshold=args.overlap_threshold,
            confidence_threshold=args.confidence_threshold,
            sample_size=None,
            seed=args.seed,
            auto_orient=args.auto_orient,
            rotate_pixels=args.rotate_pixels,
            add_border=args.add_border
        )
    else:
        # Process directory
        input_dir = os.path.join(script_dir, args.input)
        output_dir = os.path.join(script_dir, args.output, "crops")
        vis_dir = os.path.join(script_dir, args.output, "visualizations")
        
        # Check if input directory exists
        if not os.path.isdir(input_dir):
            console.print(f"{EMOJI_ERROR} [bold red]Error: Input directory '{input_dir}' not found.[/bold red]")
            return
        
        # Process images
        process_images(
            detector=detector,
            input_dir=input_dir,
            output_dir=output_dir,
            vis_dir=vis_dir,
            text_prompt=args.text_prompt,
            preserve_structure=args.preserve_structure,
            remove_overlaps=args.remove_overlaps,
            overlap_threshold=args.overlap_threshold,
            confidence_threshold=args.confidence_threshold,
            sample_size=args.sample_size,
            seed=args.seed,
            auto_orient=args.auto_orient,
            rotate_pixels=args.rotate_pixels,
            add_border=args.add_border
        )


if __name__ == "__main__":
    main() 
