import os
import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_specific_question_type(benchmark_path, question_type, images_dir="benchmark_images"):
    """Download images only for a specific question type with proper folder structure"""
    # Load benchmark
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)
    
    # Filter questions by the specified type
    questions = [q for q in benchmark['questions'] if q['question_type'] == question_type]
    
    if not questions:
        print(f"No questions found with type: {question_type}")
        return
    
    print(f"Found {len(questions)} questions of type: {question_type}")
    
    # Create the question type directory
    type_dir = os.path.join(images_dir, question_type)
    os.makedirs(type_dir, exist_ok=True)
    
    # Track successful downloads
    total_downloads = 0
    
    # Process each question
    for question in tqdm(questions, desc=f"Processing {question_type} questions"):
        q_id = question['question_id']
        
        # Create directory for this specific question
        question_dir = os.path.join(type_dir, f"{q_id}")
        os.makedirs(question_dir, exist_ok=True)
        
        # Get image URLs
        image_urls = []
        
        if 'artifact' in question:
            image_urls.append(question['artifact']['image_url'])
            if question['artifact'].get('alt_image_url'):
                image_urls.append(question['artifact']['alt_image_url'])
        elif 'artifacts' in question:
            for artifact in question['artifacts']:
                image_urls.append(artifact['image_url'])
                if artifact.get('alt_image_url'):
                    image_urls.append(artifact['alt_image_url'])
        elif 'image_urls' in question:
            image_urls.extend(question['image_urls'])
        
        # Download images for this question
        for j, url in enumerate(image_urls):
            if not url:
                continue
                
            # Get filename and path
            filename = f"image_{j+1}_{url.split('/')[-1]}"
            filepath = os.path.join(question_dir, filename)
            
            # Skip if already exists
            if os.path.exists(filepath):
                continue
                
            try:
                response = requests.get(url, timeout=10, verify=False)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img.save(filepath)
                total_downloads += 1
                print(f"Downloaded: {question_type}/{q_id}/{filename}")
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
    
    print(f"Successfully downloaded {total_downloads} images for {question_type}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download images for specific question type')
    parser.add_argument('--benchmark', type=str, required=True, 
                        help='Path to the benchmark JSON file')
    parser.add_argument('--question_type', type=str, default="chronological_sequence",
                        help='Question type to download images for (default: chronological_sequence)')
    parser.add_argument('--images_dir', type=str, default="benchmark_images",
                        help='Directory to store downloaded images')
    
    args = parser.parse_args()
    download_specific_question_type(args.benchmark, args.question_type, args.images_dir)