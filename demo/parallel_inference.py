import os
import json
import torch
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from PIL import Image

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.doc_utils import load_images_from_pdf
import sys
sys.path.append('..')
from pdf_reconstructor import PDFReconstructor


class ParallelInference:
    """
    Parallel inference class for processing multiple PDF pages using Hugging Face model.
    Supports batch processing of up to 4 pages at once.
    """
    
    def __init__(self, model_path: str = "./weights/DotsOCR", max_batch_size: int = 4):
        """
        Initialize the parallel inference class.
        
        Args:
            model_path: Path to the DotsOCR model weights
            max_batch_size: Maximum number of pages to process in one batch (default: 4)
        """
        self.max_batch_size = max_batch_size
        self.model_path = model_path
        
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'
        print("Model loaded successfully!")
    
    def inference_batch(self, images: List[Image.Image], prompt: str) -> List[str]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of PIL Image objects
            prompt: Prompt text for the model
            
        Returns:
            List of output texts for each image
        """
        if not images:
            return []
        
        # Prepare messages for each image
        messages_list = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            messages_list.append(messages)
        
        # Process all images in the batch
        all_texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in messages_list:
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            all_texts.append(text)
            all_image_inputs.extend(image_inputs)
            if video_inputs is not None:
                all_video_inputs.extend(video_inputs)

        if all_video_inputs == []:
            print("all_video_inputs is empty")
            all_video_inputs = None

        # Prepare inputs for batch processing
        inputs = self.processor(
            text=all_texts,
            images=all_image_inputs,
            videos=all_video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to("cuda")
        
        # Run inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_texts
    
    def save_text_files(self, results: List[Dict[str, Any]], save_dir: Path, pdf_name: str, 
                       save_individual_pages: bool = True, save_combined: bool = True):
        """
        Save extracted text to text files.
        
        Args:
            results: List of page results
            save_dir: Directory to save text files
            pdf_name: Name of the PDF file
            save_individual_pages: Whether to save individual page text files
            save_combined: Whether to save combined text file
        """
        if save_individual_pages:
            # Save individual page text files
            for result in results:
                page_num = result['page_no']
                output_text = result['output_text']
                
                # Save as .txt file
                text_file = save_dir / f"page_{page_num:03d}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"Page {page_num} from {pdf_name}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(output_text)
                    f.write("\n")
                
                print(f"  Saved text file: {text_file}")
        
        if save_combined:
            # Save combined text file for all pages
            combined_text_file = save_dir / f"{pdf_name}_combined.txt"
            with open(combined_text_file, 'w', encoding='utf-8') as f:
                f.write(f"Combined text from {pdf_name}\n")
                f.write("=" * 50 + "\n\n")
                
                for result in results:
                    page_num = result['page_no']
                    output_text = result['output_text']
                    
                    f.write(f"--- Page {page_num} ---\n")
                    f.write(output_text)
                    f.write("\n\n")
            
            print(f"  Saved combined text file: {combined_text_file}")

    def process_pdf_pages(self, pdf_path: str, prompt_mode: str = "prompt_layout_all_en", 
                         output_dir: str = "./output", save_text_files: bool = True, 
                         reconstruct_pdf: bool = True) -> List[Dict[str, Any]]:
        """
        Process all pages of a PDF file using batch inference.
        
        Args:
            pdf_path: Path to the PDF file
            prompt_mode: Prompt mode to use (from dict_promptmode_to_prompt)
            output_dir: Directory to save results
            save_text_files: Whether to save extracted text to .txt files
            reconstruct_pdf: Whether to reconstruct PDF from extracted text
            
        Returns:
            List of results for each page
        """
        if prompt_mode not in dict_promptmode_to_prompt:
            raise ValueError(f"Invalid prompt_mode: {prompt_mode}. Available modes: {list(dict_promptmode_to_prompt.keys())}")
        
        prompt = dict_promptmode_to_prompt[prompt_mode]
        
        print(f"Loading PDF: {pdf_path}")
        images = load_images_from_pdf(pdf_path, dpi=200)
        total_pages = len(images)
        print(f"Loaded {total_pages} pages from PDF")
        
        # Create output directory
        pdf_name = Path(pdf_path).stem
        save_dir = Path(output_dir) / pdf_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Process pages in batches
        for batch_start in tqdm(range(0, total_pages, self.max_batch_size), 
                               desc=f"Processing {pdf_name}"):
            batch_end = min(batch_start + self.max_batch_size, total_pages)
            batch_images = images[batch_start:batch_end]
            batch_pages = list(range(batch_start, batch_end))
            
            print(f"Processing batch: pages {batch_start+1}-{batch_end} of {total_pages}")
            
            # Run inference on the batch
            output_texts = self.inference_batch(batch_images, prompt)
            
            # Process results for each page in the batch
            for i, (page_idx, output_text) in enumerate(zip(batch_pages, output_texts)):
                page_num = page_idx + 1
                
                # Save individual page result
                page_result = {
                    'page_no': page_num,
                    'pdf_path': pdf_path,
                    'prompt_mode': prompt_mode,
                    'output_text': output_text,
                    'batch_idx': batch_start // self.max_batch_size
                }
                
                # Save to JSON file
                result_file = save_dir / f"page_{page_num:03d}.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(page_result, f, ensure_ascii=False, indent=2)
                
                results.append(page_result)
                print(f"  Page {page_num}: {len(output_text)} characters")
        
        # Save combined JSON results
        combined_file = save_dir / f"{pdf_name}_all_pages.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Save text files if requested
        if save_text_files:
            print(f"\nSaving text files...")
            self.save_text_files(results, save_dir, pdf_name)
        
        # Reconstruct PDF if requested
        if reconstruct_pdf:
            print(f"\nReconstructing PDF...")
            reconstructor = PDFReconstructor()
            reconstructed_pdf_path = save_dir / f"{pdf_name}_reconstructed.pdf"
            reconstructor.reconstruct_pdf_from_results(
                results=results,
                original_pdf_path=pdf_path,
                output_pdf_path=str(reconstructed_pdf_path)
            )
        
        print(f"Processing complete! Results saved to {save_dir}")
        print(f"Combined JSON results: {combined_file}")
        
        return results


def main():
    # Hardcoded parameters
    pdf_path = "/workspace/dots.ocr/gavel_new_test_set/five_merged.pdf"  # Change this to your PDF path
    model_path = "./weights/DotsOCR"
    prompt_mode = "prompt_layout_all_en"
    output_dir = "./output"
    max_batch_size = 6
    save_text_files = True
    reconstruct_pdf = True
    
    # Initialize parallel inference
    parallel_inference = ParallelInference(
        model_path=model_path,
        max_batch_size=max_batch_size
    )
    
    # Process single PDF file
    print(f"Processing PDF: {pdf_path}")
    start_time = time.time()
    results = parallel_inference.process_pdf_pages(
        pdf_path,
        prompt_mode=prompt_mode,
        output_dir=output_dir,
        save_text_files=save_text_files,
        reconstruct_pdf=reconstruct_pdf
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Processed {len(results)} pages")


if __name__ == "__main__":
    main()
