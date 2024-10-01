import gradio as gr
import cv2
import numpy as np

def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    image = np.array(image)

    # Scale the image
    # new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    # image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    #scale transform
    M_scale = np.float32([[scale, 0, 0], [0, scale, 0]])
    image = cv2.warpAffine(image, M_scale, (image.shape[1], image.shape[0]))

    center = (image.shape[1] // 2, image.shape[0] // 2)
    M_rotate = cv2.getRotationMatrix2D(center, rotation, 1)
    M_translate = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

    transformed_image = cv2.warpAffine(image, M_rotate, (image.shape[1], image.shape[0]))
    transformed_image = cv2.warpAffine(transformed_image, M_translate, (transformed_image.shape[1], transformed_image.shape[0]))

    if flip_horizontal:
        transformed_image = cv2.flip(transformed_image, 1)

    return transformed_image

def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            image_output = gr.Image(label="Transformed Image", type="numpy", shape=(400,400))

        inputs = [image_input, scale, rotation, translation_x, translation_y, flip_horizontal]

        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

interactive_transform().launch(share=True)
