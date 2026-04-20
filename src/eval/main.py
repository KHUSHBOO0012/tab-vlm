import os
import json
import time
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Local imports
from data_prep import DataPreparation
from inference import create_vlm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TemporalAnomalyEvaluator:
    def __init__(self, benchmark_file, model_name, model_id=None, output_dir="results", 
                 data_dir="data", save_images=True, use_existing_images=False):
        """Initialize the evaluator"""
        self.benchmark_file = benchmark_file
        self.model_name = model_name
        self.model_id = model_id
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.save_images = save_images
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_prep = DataPreparation(
            benchmark_file, 
            data_dir=data_dir, 
            save_images=save_images,
            use_existing_images=use_existing_images
        )
        self.vlm = create_vlm(model_name, model_id)
        self.questions = self.data_prep.questions
        self.results = []
        
        # Cost tracking for API models
        self.total_cost = 0.0
        
        logger.info(f"Initialized evaluator with {len(self.questions)} questions")
        logger.info(f"Model: {model_name} ({model_id or 'default'})")
        logger.info(f"Images will be {'saved to' if save_images else 'not saved to'} {data_dir}")
        logger.info(f"Using {'existing' if use_existing_images else 'fresh'} images")
    
    def load_existing_results(self):
        """Load existing results for checkpointing"""
        results_file = f"{self.output_dir}/detailed_results.json"
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            logger.info(f"Loaded {len(existing_results)} existing results")
            return existing_results
        return []
    
    def get_completed_question_ids(self, results):
        """Get set of completed question IDs"""
        return set(r['question_id'] for r in results if r.get('success', False))
    
    def save_checkpoint(self):
        """Save current results as checkpoint"""
        checkpoint_data = {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "total_cost": getattr(self.vlm, 'total_cost', 0.0),
            "results": self.results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{self.output_dir}/checkpoint.json", 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self):
        """Load results from checkpoint"""
        checkpoint_file = f"{self.output_dir}/checkpoint.json"
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            if (checkpoint.get('model_name') == self.model_name and 
                checkpoint.get('model_id') == self.model_id):
                self.results = checkpoint.get('results', [])
                if hasattr(self.vlm, 'total_cost'):
                    self.vlm.total_cost = checkpoint.get('total_cost', 0.0)
                logger.info(f"Resumed from checkpoint with {len(self.results)} results")
                return True
        return False
    def process_question(self, question):
        """Process a single question and return detailed result"""
        question_id = question['question_id']
        question_type = question['question_type']
        
        logger.info(f"Processing question {question_id} ({question_type})")
        
        try:
            # Download images
            images_with_ids = self.data_prep.download_images(question)
            
            # Prepare prompt
            prompt = self.data_prep.prepare_prompt(question)
            
            # Run inference
            response = self.vlm.run_inference(prompt, images_with_ids)
            
            if not response:
                return {
                    "question_id": question_id,
                    "question_type": question_type,
                    "success": False,
                    "error": "No response from model",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
            # Extract answer
            model_answer = self.vlm.extract_answer(response, question_type)
            
            # Get correct answer based on question type
            if question_type in ['odd_one_out_period', 'style_period_attribution']:
                correct_answer = question['correct_index']
                is_correct = model_answer == correct_answer if model_answer is not None else False
            elif question_type in ['manufacturing_technique', 'material_availability']:
                correct_answer = question['correct_indices']
                is_correct = set(model_answer) == set(correct_answer) if model_answer is not None else False
            elif question_type == 'period_grouping':
                correct_answer = question['correct_indices']
                is_correct = set(model_answer) == set(correct_answer) if model_answer is not None else False
            elif question_type == 'chronological_sequence':
                correct_answer = question['correct_sequence']
                is_correct = model_answer == correct_answer if model_answer is not None else False
            else:
                correct_answer = None
                is_correct = None
            
            # Prepare image info for logging
            image_info = []
            for img, img_id, title, img_path, img_url in images_with_ids:
                image_info.append({
                    "image_id": img_id,
                    "image_title": title,
                    "image_size": f"{img.width}x{img.height}",
                    "image_path": img_path,
                    "image_url": img_url
                })
            
            return {
                "question_id": question_id,
                "question_type": question_type,
                "question_text": question['question_text'],
                "prompt": prompt,
                "image_info": image_info,
                "model_response": response,
                "model_answer": model_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "success": True,
                "current_cost": getattr(self.vlm, 'total_cost', 0.0),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            return {
                "question_id": question_id,
                "question_type": question_type,
                "success": False,
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def evaluate_all(self, sample_size=None, resume=True):
        """Evaluate all questions with checkpointing"""
        # Load existing results if resuming
        if resume and self.load_checkpoint():
            completed_ids = self.get_completed_question_ids(self.results)
            logger.info(f"Resuming evaluation - {len(completed_ids)} questions already completed")
        else:
            completed_ids = set()
        
        # Filter questions
        questions_to_evaluate = self.questions
        if sample_size and sample_size < len(questions_to_evaluate):
            import random
            questions_to_evaluate = random.sample(questions_to_evaluate, sample_size)
            logger.info(f"Sampling {sample_size} questions for evaluation")
        
        # Filter out completed questions
        remaining_questions = [q for q in questions_to_evaluate if q['question_id'] not in completed_ids]
        
        if not remaining_questions:
            logger.info("All questions already completed!")
            self.calculate_metrics()
            return
        
        logger.info(f"Processing {len(remaining_questions)} remaining questions")
        
        start_time = time.time()
        
        # Process each question
        for i, question in enumerate(tqdm(remaining_questions, desc="Evaluating questions")):
            result = self.process_question(question)
            self.results.append(result)
            
            # Save checkpoint every 5 questions
            if (i + 1) % 5 == 0:
                self.save_checkpoint()
                self.save_results()
                
                # Log progress with cost info
                success_count = sum(1 for r in self.results if r.get('success', False))
                current_cost = getattr(self.vlm, 'total_cost', 0.0)
                
                logger.info(f"Progress: {i + 1}/{len(remaining_questions)} questions")
                logger.info(f"Success rate: {success_count}/{len(self.results)} ({success_count/len(self.results):.1%})")
                if current_cost > 0:
                    logger.info(f"Current cost: ${current_cost:.6f}")
        
        # Final save
        self.save_checkpoint()
        self.save_results()
        
        # Calculate metrics
        self.calculate_metrics()
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation completed in {total_time:.2f} seconds")
        logger.info(f"Average time per question: {total_time/len(remaining_questions):.2f} seconds")
        
        # Final cost summary
        final_cost = getattr(self.vlm, 'total_cost', 0.0)
        if final_cost > 0:
            logger.info(f"Total cost: ${final_cost:.6f}")
            logger.info(f"Average cost per question: ${final_cost/len(self.results):.6f}")
    
    def save_results(self):
        """Save detailed results to JSON and CSV"""
        # Save detailed JSON
        results_with_summary = {
            "model_info": {
                "model_name": self.model_name,
                "model_id": self.model_id,
                "total_cost": getattr(self.vlm, 'total_cost', 0.0),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "summary": {
                "total_questions": len(self.results),
                "successful_questions": sum(1 for r in self.results if r.get('success', False)),
                "unsuccessful_questions": sum(1 for r in self.results if not r.get('success', False)),
                "success_rate": sum(1 for r in self.results if r.get('success', False)) / len(self.results) if self.results else 0
            },
            "results": self.results
        }
        
        with open(f"{self.output_dir}/detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_with_summary, f, indent=2, ensure_ascii=False)
        
        # Save CSV for easy analysis
        csv_data = []
        for result in self.results:
            if result.get('success', False):
                # Flatten image info
                image_ids = [img['image_id'] for img in result.get('image_info', [])]
                image_titles = [img['image_title'] for img in result.get('image_info', [])]
                image_paths = [img.get('image_path', 'N/A') for img in result.get('image_info', [])]
                image_urls = [img.get('image_url', 'N/A') for img in result.get('image_info', [])]
                
                csv_data.append({
                    'question_id': result['question_id'],
                    'question_type': result['question_type'],
                    'question_text': result['question_text'][:100] + "..." if len(result['question_text']) > 100 else result['question_text'],
                    'image_ids': "; ".join(image_ids),
                    'image_titles': "; ".join(image_titles),
                    'image_paths': "; ".join(image_paths),
                    'image_urls': "; ".join(image_urls),
                    'model_answer': str(result.get('model_answer', '')),
                    'correct_answer': str(result.get('correct_answer', '')),
                    'is_correct': result.get('is_correct', False),
                    'response_length': len(result.get('model_response', '')),
                    'current_cost': result.get('current_cost', 0.0),
                    'timestamp': result.get('timestamp', '')
                })
            else:
                # Include failed questions
                csv_data.append({
                    'question_id': result['question_id'],
                    'question_type': result.get('question_type', 'unknown'),
                    'question_text': 'FAILED',
                    'image_ids': 'FAILED',
                    'image_titles': 'FAILED',
                    'image_paths': 'FAILED',
                    'image_urls': 'FAILED',
                    'model_answer': 'FAILED',
                    'correct_answer': 'FAILED',
                    'is_correct': False,
                    'response_length': 0,
                    'current_cost': result.get('current_cost', 0.0),
                    'timestamp': result.get('timestamp', ''),
                    'error': result.get('error', 'Unknown error')
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(f"{self.output_dir}/results_summary.csv", index=False)
    
    def calculate_metrics(self):
        """Calculate and save detailed evaluation metrics"""
        total_questions = len(self.results)
        successful_results = [r for r in self.results if r.get('success', False)]
        failed_results = [r for r in self.results if not r.get('success', False)]
        
        if not successful_results:
            logger.warning("No successful results to calculate metrics")
            return
        
        # Overall metrics
        success_rate = len(successful_results) / total_questions
        correct_count = sum(1 for r in successful_results if r.get('is_correct', False))
        overall_accuracy = correct_count / len(successful_results)
        
        # Metrics by question type
        type_metrics = {}
        question_types = set(r.get('question_type', 'unknown') for r in successful_results)
        
        for q_type in question_types:
            type_results = [r for r in successful_results if r.get('question_type') == q_type]
            type_failed = [r for r in failed_results if r.get('question_type') == q_type]
            type_correct = sum(1 for r in type_results if r.get('is_correct', False))
            type_accuracy = type_correct / len(type_results) if type_results else 0
            
            type_metrics[q_type] = {
                'total_attempted': len(type_results) + len(type_failed),
                'successful_count': len(type_results),
                'failed_count': len(type_failed),
                'correct_count': type_correct,
                'accuracy': type_accuracy,
                'success_rate': len(type_results) / (len(type_results) + len(type_failed)) if (len(type_results) + len(type_failed)) > 0 else 0
            }
        
        # Cost analysis
        final_cost = getattr(self.vlm, 'total_cost', 0.0)
        cost_per_question = final_cost / total_questions if total_questions > 0 else 0
        cost_per_successful = final_cost / len(successful_results) if successful_results and final_cost > 0 else 0
        
        # Error analysis
        error_types = {}
        for result in failed_results:
            error = result.get('error', 'Unknown error')
            error_types[error] = error_types.get(error, 0) + 1
        
        # Create comprehensive metrics
        metrics = {
            'model_info': {
                'model_name': self.model_name,
                'model_id': self.model_id,
                'evaluation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'overall_performance': {
                'total_questions': total_questions,
                'successful_evaluations': len(successful_results),
                'failed_evaluations': len(failed_results),
                'success_rate': success_rate,
                'overall_accuracy': overall_accuracy,
                'correct_answers': correct_count
            },
            'cost_analysis': {
                'total_cost': final_cost,
                'cost_per_question': cost_per_question,
                'cost_per_successful_question': cost_per_successful,
                'currency': 'USD'
            },
            'performance_by_type': type_metrics,
            'error_analysis': {
                'total_errors': len(failed_results),
                'error_types': error_types
            }
        }
        
        # Save metrics
        with open(f"{self.output_dir}/comprehensive_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Print comprehensive summary
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name} ({self.model_id or 'default'})")
        logger.info(f"Total questions: {total_questions}")
        logger.info(f"Successful evaluations: {len(successful_results)} ({success_rate:.1%})")
        logger.info(f"Failed evaluations: {len(failed_results)} ({len(failed_results)/total_questions:.1%})")
        logger.info(f"Overall accuracy: {overall_accuracy:.2%} ({correct_count}/{len(successful_results)})")
        
        if final_cost > 0:
            logger.info(f"\nCost Analysis:")
            logger.info(f"Total cost: ${final_cost:.6f}")
            logger.info(f"Cost per question: ${cost_per_question:.6f}")
            logger.info(f"Cost per successful question: ${cost_per_successful:.6f}")
        
        logger.info(f"\nPerformance by Question Type:")
        for q_type, stats in type_metrics.items():
            logger.info(f"  {q_type}:")
            logger.info(f"    Success rate: {stats['success_rate']:.1%} ({stats['successful_count']}/{stats['total_attempted']})")
            logger.info(f"    Accuracy: {stats['accuracy']:.2%} ({stats['correct_count']}/{stats['successful_count']})")
        
        if error_types:
            logger.info(f"\nError Analysis:")
            for error, count in error_types.items():
                logger.info(f"  {error}: {count} occurrences")
        
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='VLM evaluation for Cultural Anachronism and Temporal Reasoning benchmark')
    parser.add_argument('--benchmark', type=str, required=True,
                        help='Path to the benchmark JSON file')
    parser.add_argument('--model', type=str, required=True,
                        choices=['gpt4o', 'gpt4o-mini', 'claude', 'claude4', 'qwen', 'internvl', 'deepseek', 'gemma3'],
                        help='Model to evaluate')
    parser.add_argument('--model_id', type=str, default=None,
                        help='Specific model ID (optional)')
    parser.add_argument('--output_dir', type=str, default="results",
                        help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default="data",
                        help='Directory to save images')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of questions to sample (for testing)')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='Save images to disk (default: True)')
    parser.add_argument('--no_save_images', action='store_true',
                        help='Do not save images to disk')
    parser.add_argument('--use_existing_images', action='store_true',
                        help='Use existing images if available')
    parser.add_argument('--no_resume', action='store_true',
                        help='Start fresh instead of resuming')
    
    args = parser.parse_args()
    
    # Handle image saving flag
    if args.no_save_images:
        args.save_images = False
    
    # Check API keys for API-based models
    if args.model in ['gpt4o', 'gpt4o-mini'] and not os.getenv('OPENAI_API_KEY'):
        logger.error("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return
    
    if args.model in ['claude', 'claude4'] and not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
        return
    
    # Create evaluator
    evaluator = TemporalAnomalyEvaluator(
        args.benchmark,
        args.model,
        args.model_id,
        args.output_dir,
        args.data_dir,
        args.save_images,
        args.use_existing_images
    )
    
    # Run evaluation
    evaluator.evaluate_all(args.sample_size, resume=not args.no_resume)


if __name__ == "__main__":
    main()