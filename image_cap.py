"""
Image Captioning Web App

This script uses a trained model (BLIP) to generate captions for images using Gradio.
"""

import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

print("Loading processor and model...")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Done.")


def caption_image(input_image: np.ndarray):
    """
    Generate a caption for the input image.

    Parameters:
    input_image (np.ndarray): Input image as a NumPy array.

    Returns:
    tuple: A tuple containing a static string "The image caption is: " and the generated caption.
    """
    # Load image and convert to RGB
    image = Image.fromarray(input_image).convert('RGB')

    # Question for image captioning and generate inputs
    text = 'the image of'
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)  # A sequence of tokens

    # Decode the generated tokens (outputs) to human-readable text
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return "The image caption is: ", caption


gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions "
                "for images using a trained model (BLIP)."
).launch()
