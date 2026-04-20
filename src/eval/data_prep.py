import os
import json
import requests
from PIL import Image
from io import BytesIO
import logging
import hashlib

logger = logging.getLogger(__name__)

class DataPreparation:
    """Data preparation with image saving and caching"""
    
    def __init__(self, benchmark_file, data_dir="data", save_images=True, use_existing_images=False):
        self.benchmark_file = benchmark_file
        self.data_dir = data_dir
        self.save_images = save_images
        self.use_existing_images = use_existing_images
        
        # Create data directory
        if save_images:
            os.makedirs(data_dir, exist_ok=True)
            
        self.load_benchmark()
    
    def load_benchmark(self):
        """Load benchmark data from file"""
        with open(self.benchmark_file, 'r', encoding='utf-8') as f:
            self.benchmark = json.load(f)
        self.questions = self.benchmark['questions']
        logger.info(f"Loaded {len(self.questions)} questions from benchmark")
    
    def get_image_path(self, url):
        """Generate a consistent file path for an image URL"""
        # Create a hash of the URL for consistent naming
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.data_dir, f"{url_hash}.jpg")
    
    def fetch_image(self, url):
        """Fetch an image from URL, with caching if save_images is enabled"""
        if self.save_images:
            image_path = self.get_image_path(url)
            
            # Check if image already exists
            if self.use_existing_images and os.path.exists(image_path):
                logger.debug(f"Using cached image: {image_path}")
                return Image.open(image_path).convert('RGB'), image_path
        
        # Download image
        verify = 'museumsofindia.gov.in' not in url
        response = requests.get(url, timeout=10, verify=verify)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Save image if enabled
        if self.save_images:
            image_path = self.get_image_path(url)
            img.save(image_path, 'JPEG', quality=95)
            logger.debug(f"Saved image: {image_path}")
            return img, image_path
        
        return img, None
    
    def download_images(self, question):
        """Download all images for a question and return PIL Images with IDs and paths"""
        question_type = question['question_type']
        images_with_ids = []
        
        if question_type in ['manufacturing_technique', 'material_availability', 'style_period_attribution']:
            # Single image questions
            artifact = question['artifact']
            img, img_path = self.fetch_image(artifact['image_url'])
            image_id = f"q{question['question_id']}_img0"
            images_with_ids.append((img, image_id, artifact.get('title', 'Unknown'), img_path, artifact['image_url']))
        else:
            # Multiple image questions
            for i, artifact in enumerate(question['artifacts']):
                img, img_path = self.fetch_image(artifact['image_url'])
                image_id = f"q{question['question_id']}_img{i}"
                images_with_ids.append((img, image_id, artifact.get('title', 'Unknown'), img_path, artifact['image_url']))
        
        return images_with_ids
    
    def prepare_prompt(self, question):
        """Prepare prompt based on question type"""
        question_type = question['question_type']
        
        if question_type == 'odd_one_out_period':
            return f"""Below are 4 historical artifacts. One of them belongs to a different time period than the others.

Question: {question['question_text']}

For the artifact that belongs to a different time period than the others:
Identify which one it is by number (1, 2, 3, or 4)
Explain why you think it belongs to a different period

Your answer must start with the number of the artifact that is different."""

        elif question_type in ['manufacturing_technique', 'material_availability']:
            options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(question['options'])])
            return f"""Look at this historical artifact and answer the following question:

Question: {question['question_text']}

Please select all options that apply from the following choices:
{options_text}

For your answer, simply list the numbers of all correct options (e.g., "1, 3" if options 1 and 3 are correct)."""

        elif question_type == 'period_grouping':
            return f"""Below are 5 historical artifacts. Three of them were likely created during the same time period.

Question: {question['question_text']}

For the three artifacts that were likely created during the same time period:
Identify which ones they are by number (e.g., "1, 3, 5")
Explain why you think these three belong to the same period

Your answer must start with the numbers of the three artifacts from the same period, separated by commas."""

        elif question_type == 'chronological_sequence':
            return f"""Below are 4 historical artifacts from different time periods.

Question: {question['question_text']}

Please provide the sequence from oldest to newest by listing the artifact numbers (1-4) in chronological order.

Your answer must be in the format: "1, 3, 2, 4" (where these numbers represent the order from oldest to newest)."""

        elif question_type == 'style_period_attribution':
            options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(question['options'])])
            return f"""Look at this historical artifact and answer the following question:

Question: {question['question_text']}

Please select the correct option from the following choices:
{options_text}

For your answer, simply provide the number of the correct option (1, 2, 3, or 4)."""

        return f"Question: {question['question_text']}"