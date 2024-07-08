"""
Web Scraping and Image Captioning from web pages

This script scrapes images from a web page
and generates captions using a pretrained model (BLIP).
"""

from io import BytesIO
import requests
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image

# Load processor and model
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Done.")

# URL of the page to scrape
URL = "https://en.wikipedia.org/wiki/IBM"

try:
    # Download the page
    response = requests.get(URL, timeout=10)  # Added timeout argument to requests.get
    response.raise_for_status()  # Raise an error for bad status codes

    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all image elements
    img_elements = soup.find_all("img")

    # Open a file to write the captions, Specified encoding for open()
    with open('captions.txt', 'w', encoding='utf-8') as captions_file:
        # Iterate over each img element
        for img_element in img_elements:
            img_url = img_element.get('src')

            # Skip if the image is an SVG or too small (likely an icon)
            if '.svg' in img_url or '1x1' in img_url:
                continue

            # Correct the URL if it's malformed
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif not img_url.startswith(('http://', 'https://')):
                continue  # Skip URLs that don't start with http:// or https://

            try:
                # Download the image with timeout
                response = requests.get(img_url, timeout=10)
                response.raise_for_status()  # Raise an error for bad status codes

                # Convert the image data to a PIL Image
                raw_image = Image.open(BytesIO(response.content))

                # Skip very small images
                if raw_image.size[0] * raw_image.size[1] < 1:
                    continue

                raw_image = raw_image.convert('RGB')

                # Process the image
                inputs = processor(images=raw_image, return_tensors='pt')

                # Generate a caption for the image
                outputs = model.generate(**inputs, max_length=50)

                # Decode the generated tokens to text
                caption = processor.decode(outputs[0], skip_special_tokens=True)

                # Write the caption to the file, prepended by the image URL
                captions_file.write(f"{img_url}: {caption}\n")

            except requests.exceptions.RequestException as e:
                print(f"Error processing image {img_url}: {e}")
                continue

            except (IOError, ValueError) as e:
                print(f"Error processing image {img_url}: {e}")
                continue

except requests.exceptions.RequestException as e:
    print(f"Error downloading page {URL}: {e}")
