import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

def load_img(image, image_size=(256, 256)):
    """Load and preprocess image from PIL Image"""
    # Convert PIL to tensor
    img = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
    img = img / 255.0  # Normalize to [0,1]
    img = img[tf.newaxis, ...]  # Add batch dimension
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def crop_center(image):
    """Crop image to square from center"""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    return tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)

# Load the model once at startup
print("Loading style transfer model...")
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
print("Model loaded successfully!")

def stylize_images(content_image, style_image):
    """Apply style transfer to images"""
    try:
        # Process content image
        content_image = load_img(content_image, (384, 384))
        content_image = crop_center(content_image)
        
        # Process style image  
        style_image = load_img(style_image, (256, 256))
        style_image = crop_center(style_image)
        style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
        
        # Apply style transfer
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        
        # Convert back to PIL Image
        stylized_image = tf.squeeze(stylized_image, 0)  # Remove batch dimension
        stylized_image = tf.clip_by_value(stylized_image, 0, 1)  # Ensure valid range
        stylized_image = tf.cast(stylized_image * 255, tf.uint8)  # Convert to uint8
        stylized_array = stylized_image.numpy()
        result_image = Image.fromarray(stylized_array)
        
        return result_image
        
    except Exception as e:
        print(f"Error during style transfer: {e}")
        return None

# Create Gradio interface
with gr.Blocks(title="Neural Style Transfer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 Neural Style Transfer
        Upload a content image and a style image to create artistic combinations!
        
        **How to use:**
        1. Upload your content image (the photo you want to stylize)
        2. Upload your style image (the artwork style you want to apply)  
        3. Click "Generate Stylized Image"
        """
    )
    
    with gr.Row():
        with gr.Column():
            content_input = gr.Image(
                label="Content Image", 
                type="pil",
                height=300
            )
            style_input = gr.Image(
                label="Style Image", 
                type="pil", 
                height=300
            )
            generate_btn = gr.Button("🎨 Generate Stylized Image", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(
                label="Stylized Result", 
                height=400
            )
    
    # Example images
    gr.Markdown("### 📸 Try these examples:")
    gr.Examples(
        examples=[
            ["examples/content1.jpg", "examples/style1.jpg"],
            ["examples/content2.jpg", "examples/style2.jpg"],
        ],
        inputs=[content_input, style_input],
        outputs=output_image,
        fn=stylize_images,
        cache_examples=False
    )
    
    generate_btn.click(
        fn=stylize_images,
        inputs=[content_input, style_input],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch()
