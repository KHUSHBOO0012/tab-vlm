import os
import re
import base64
import torch
from io import BytesIO
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class VLMInference(ABC):
    """Base class for vision-language model inference"""
    
    def __init__(self, model_id, model_name):
        self.model_id = model_id
        self.model_name = model_name
        self.load_model()
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def run_inference(self, prompt, images_with_ids):
        pass
    
    def extract_answer(self, response, question_type):
        """Extract structured answers from model responses"""
        if not response:
            return None
            
        response_clean = response.strip()
        
        if question_type == 'odd_one_out_period':
            # Look for number at start of response
            first_line_match = re.match(r'^(\d+)', response_clean)
            if first_line_match:
                num = int(first_line_match.group(1))
                if 1 <= num <= 4:
                    return num - 1  # Convert to 0-based index
            
            # Look for patterns like "Artifact 2", "2)", "2."
            patterns = [r'artifact\s*(\d+)', r'(?:^|\s)(\d+)(?:\)|\.|\s|$)']
            for pattern in patterns:
                matches = re.findall(pattern, response_clean.lower())
                if matches:
                    for match in matches:
                        num = int(match)
                        if 1 <= num <= 4:
                            return num - 1
            return None
            
        elif question_type in ['manufacturing_technique', 'material_availability']:
            # Look for comma-separated numbers
            numbers = re.findall(r'[1-4]', response_clean[:100])
            found_numbers = []
            for num_str in numbers:
                num = int(num_str)
                if 1 <= num <= 4 and (num - 1) not in found_numbers:
                    found_numbers.append(num - 1)
            return found_numbers
            
        elif question_type == 'period_grouping':
            # Look for three numbers 1-5
            numbers = re.findall(r'[1-5]', response_clean[:200])
            found_numbers = []
            for num_str in numbers:
                num = int(num_str)
                if 1 <= num <= 5 and (num - 1) not in found_numbers:
                    found_numbers.append(num - 1)
                    if len(found_numbers) >= 3:
                        break
            return found_numbers[:3] if len(found_numbers) >= 3 else []
            
        elif question_type == 'chronological_sequence':
            # Look for sequence of 4 numbers
            sequence_match = re.search(r'([1-4]\s*,\s*[1-4]\s*,\s*[1-4]\s*,\s*[1-4])', response_clean)
            if sequence_match:
                numbers = re.findall(r'[1-4]', sequence_match.group(1))
                if len(numbers) == 4:
                    return [int(n) - 1 for n in numbers]
            
            # Fallback: any 4 numbers 1-4
            numbers = re.findall(r'[1-4]', response_clean[:100])
            if len(numbers) >= 4:
                return [int(n) - 1 for n in numbers[:4]]
            return []
            
        elif question_type == 'style_period_attribution':
            # Look for single number
            patterns = [r'(?:option|choice|answer)(?:\s+is)?:?\s*([1-4])', r'(?:^|\s)([1-4])(?:\)|\.|\s|$)']
            for pattern in patterns:
                matches = re.findall(pattern, response_clean.lower())
                if matches:
                    for match in matches:
                        num = int(match)
                        if 1 <= num <= 4:
                            return num - 1
            return None
            
        return None


class OpenAIVision(VLMInference):
    """OpenAI GPT-4o and GPT-4o mini inference"""
    
    def __init__(self, model_id="gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.total_cost = 0.0
        self.pricing = {
            "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
            "gpt-4o-mini": {"input_per_1k": 0.000150, "output_per_1k": 0.000600}
        }
        super().__init__(model_id, "openai")
    
    def load_model(self):
        pass  # API-based, no loading needed
    
    def image_to_base64(self, image):
        """Convert PIL image to base64"""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def run_inference(self, prompt, images_with_ids):
        """Run inference with OpenAI API"""
        content = [{"type": "text", "text": prompt}]
        
        # Add images
        for image, image_id, title, img_path, img_url in images_with_ids:
            img_base64 = self.image_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                    "detail": "high"
                }
            })
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=512,
            temperature=0.0
        )
        
        # Track cost
        usage = response.usage
        if self.model_id in self.pricing:
            pricing = self.pricing[self.model_id]
            cost = (usage.prompt_tokens / 1000) * pricing["input_per_1k"] + \
                   (usage.completion_tokens / 1000) * pricing["output_per_1k"]
            self.total_cost += cost
            logger.info(f"API cost: ${cost:.6f} (Total: ${self.total_cost:.6f})")
        
        return response.choices[0].message.content


class AnthropicVision(VLMInference):
    """Anthropic Claude Sonnet inference"""
    
    def __init__(self, model_id="claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.total_cost = 0.0
        # Updated pricing based on official rates (per 1k tokens)
        self.pricing = {
            "claude-3-5-sonnet-20241022": {"input_per_1k": 0.003, "output_per_1k": 0.015},
            "claude-sonnet-4-20250514": {"input_per_1k": 0.003, "output_per_1k": 0.015},
            "claude-3-opus-20240229": {"input_per_1k": 0.015, "output_per_1k": 0.075},
            "claude-3-haiku-20240307": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
            "claude-3-5-haiku-20241022": {"input_per_1k": 0.0008, "output_per_1k": 0.004}
        }
        super().__init__(model_id, "anthropic")
    
    def load_model(self):
        pass  # API-based, no loading needed
    
    def image_to_base64(self, image):
        """Convert PIL image to base64"""
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def run_inference(self, prompt, images_with_ids):
        """Run inference with Anthropic API"""
        content = []
        
        # Add images first
        for image, image_id, title, img_path, img_url in images_with_ids:
            img_base64 = self.image_to_base64(image)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_base64
                }
            })
        
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": content}]
        )
        
        # Track cost (approximate)
        if self.model_id in self.pricing:
            pricing = self.pricing[self.model_id]
            # Estimate tokens (rough approximation)
            input_tokens = len(prompt.split()) * 1.3 + len(images_with_ids) * 100
            output_tokens = len(response.content[0].text.split()) * 1.3
            cost = (input_tokens / 1000) * pricing["input_per_1k"] + \
                   (output_tokens / 1000) * pricing["output_per_1k"]
            self.total_cost += cost
            logger.info(f"API cost (est): ${cost:.6f} (Total: ${self.total_cost:.6f})")
        
        return response.content[0].text


class QwenVLInference(VLMInference):
    """Qwen VL model inference"""
    
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        super().__init__(model_id, "qwen")
    
    def load_model(self):
        """Load Qwen VL model"""
        from transformers import AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        logger.info(f"Loaded Qwen model: {self.model_id}")
    
    def run_inference(self, prompt, images_with_ids):
        """Run inference with Qwen VL"""
        images = [img for img, _, _, _, _ in images_with_ids]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant with expertise in art history and archaeology."},
            {"role": "user", "content": []}
        ]
        
        # Add prompt text first
        messages[1]["content"].append({"type": "text", "text": prompt})
        
        # Add images
        for img in images:
            messages[1]["content"].append({
                "type": "image", 
                "image": img,
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 1280 * 28 * 28
            })

        from qwen_vl_utils import process_vision_info
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1
            )
        
        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response


class InternVLInference(VLMInference):
    """InternVL model inference"""
    
    def __init__(self, model_id="OpenGVLab/InternVL3-2B"):
        super().__init__(model_id, "internvl")
    
    def load_model(self):
        """Load InternVL model"""
        from transformers import AutoModel, AutoTokenizer
        
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            use_fast=False
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Loaded InternVL model: {self.model_id}")
    
    def load_image(self, image, input_size=448):
        """Process image for InternVL"""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Simple resize instead of complex dynamic preprocessing
        image = image.resize((input_size, input_size))
        pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
        
        return pixel_values
    
    def run_inference(self, prompt, images_with_ids):
        """Run inference with InternVL"""
        if len(images_with_ids) == 1:
            # Single image
            image = images_with_ids[0][0]
            pixel_values = self.load_image(image).to(torch.bfloat16).cuda()
            question = '<image>\n' + prompt
        else:
            # Multiple images
            pixel_values_list = []
            for img, _, _, _, _ in images_with_ids:
                pixel_values = self.load_image(img).to(torch.bfloat16).cuda()
                pixel_values_list.append(pixel_values)
            
            pixel_values = torch.cat(pixel_values_list, dim=0)
            image_labels = ''.join([f'Image-{i+1}: <image>\n' for i in range(len(images_with_ids))])
            question = image_labels + prompt
        
        generation_config = dict(max_new_tokens=512, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        
        return response


class DeepSeekVL2Inference(VLMInference):
    """DeepSeek-VL2 model inference"""
    
    def __init__(self, model_id="deepseek-ai/deepseek-vl2-tiny"):
        super().__init__(model_id, "deepseek-vl2")
    
    def load_model(self):
        """Load DeepSeek-VL2 model"""
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        ).to(torch.bfloat16).cuda().eval()
        
        logger.info(f"Loaded DeepSeek-VL2 model: {self.model_id}")
    
    def run_inference(self, prompt, images_with_ids):
        """Run inference with DeepSeek-VL2"""
        from deepseek_vl2.utils.io import load_pil_images
        
        if len(images_with_ids) == 1:
            # Single image
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{prompt}",
                    "images": [images_with_ids[0][0]],  # PIL image
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            # Multiple images - use image placeholders
            image_placeholders = "".join(["<image_placeholder>" for _ in images_with_ids])
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"{image_placeholders}{prompt}",
                    "images": [img for img, _, _, _, _ in images_with_ids],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        
        # Load images and prepare inputs
        pil_images = [img for img, _, _, _, _ in images_with_ids]
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)
        
        # Run image encoder to get embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer


class Gemma3Vision(VLMInference):
    """Gemma-3 Vision model inference"""
    
    def __init__(self, model_id="google/gemma-3-4b-it"):
        super().__init__(model_id, "gemma3")
    
    def load_model(self):
        """Load Gemma-3 model"""
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto"
        ).eval()
        
        logger.info(f"Loaded Gemma-3 model: {self.model_id}")
    
    def run_inference(self, prompt, images_with_ids):
        """Run inference with Gemma-3"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant with expertise in art history and archaeology."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]
        
        # Add images first
        for image, _, _, _, _ in images_with_ids:
            messages[1]["content"].append({"type": "image", "image": image})
        
        # Add text prompt
        messages[1]["content"].append({"type": "text", "text": prompt})
        
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False
            )
            generation = generation[0][input_len:]
        
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded


def create_vlm(model_name, model_id=None):
    """Factory function to create VLM inference objects"""
    if model_name.lower() in ["gpt4o", "openai"]:
        return OpenAIVision(model_id or "gpt-4o")
    elif model_name.lower() in ["gpt4o-mini", "openai-mini"]:
        return OpenAIVision("gpt-4o-mini")
    elif model_name.lower() in ["claude", "anthropic"]:
        return AnthropicVision(model_id or "claude-3-5-sonnet-20241022")
    elif model_name.lower() in ["claude4", "claude-4"]:
        return AnthropicVision("claude-sonnet-4-20250514")
    elif model_name.lower() == "qwen":
        return QwenVLInference(model_id or "Qwen/Qwen2.5-VL-3B-Instruct")
    elif model_name.lower() == "internvl":
        return InternVLInference(model_id or "OpenGVLab/InternVL3-2B")
    elif model_name.lower() in ["deepseek", "deepseek-vl2"]:
        return DeepSeekVL2Inference(model_id or "deepseek-ai/deepseek-vl2-tiny")
    elif model_name.lower() in ["gemma3", "gemma-3"]:
        return Gemma3Vision(model_id or "google/gemma-3-4b-it")
    else:
        raise ValueError(f"Unsupported model: {model_name}")