from dots_ocr.parser import DotsOCRParser
from PIL import Image
import time
import json
import os
import shutil

# Initialize parser with vLLM server config
parser = DotsOCRParser(
    ip='localhost',
    port=8000,
    model_name='model',
    temperature=0.1,
    top_p=1.0,
    max_completion_tokens=24000
)

# Load your JPG image
# image = Image.open('demo/demo_image1.jpg')

# Parse with different modes:
# 1. Full layout parsing (JSON output with bboxes, categories, text)
# result = parser.parse_image(
#     input_path='demo/demo_image1.jpg',
#     filename='demo_image1',
#     prompt_mode='prompt_layout_all_en',
#     save_dir='./output'
# )

start_time = time.time()
# Parse PDF directly
results = parser.parse_pdf(
    input_path='/workspace/dots.ocr/gavel_new_test_set/sale_deed_house.pdf',
    filename='/sale_deed_house',
    prompt_mode='prompt_layout_all_en',
    save_dir='./output'
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(f"Processed {len(results)} pages")
print(f"Average time per page: {(end_time - start_time) / len(results)} seconds")

# Save results in pretty format in output2 folder
output2_dir = './output2'
os.makedirs(output2_dir, exist_ok=True)

for i, result in enumerate(results):
    # Create folder for each page
    page_folder = os.path.join(output2_dir, f'page_{i+1:03d}')
    os.makedirs(page_folder, exist_ok=True)
    
    # Save result metadata
    metadata = {
        'page_no': result.get('page_no', i),
        'input_height': result.get('input_height'),
        'input_width': result.get('input_width'),
        'file_path': result.get('file_path'),
        'filtered': result.get('filtered', False)
    }
    
    with open(os.path.join(page_folder, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Copy existing files from original output
    if 'layout_info_path' in result:
        shutil.copy2(result['layout_info_path'], os.path.join(page_folder, 'layout.json'))
    
    if 'layout_image_path' in result:
        shutil.copy2(result['layout_image_path'], os.path.join(page_folder, 'layout_visualization.jpg'))
    
    if 'md_content_path' in result:
        shutil.copy2(result['md_content_path'], os.path.join(page_folder, 'content.md'))
    
    if 'md_content_nohf_path' in result:
        shutil.copy2(result['md_content_nohf_path'], os.path.join(page_folder, 'content_no_header_footer.md'))

print(f"Results saved in organized format to {output2_dir}")
print(f"Each page has its own folder with metadata, layout info, visualization, and content files")