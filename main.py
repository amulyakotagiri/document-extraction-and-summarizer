import os
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline



app = FastAPI()

templates = Jinja2Templates(directory="templates")

# summarizer = pipeline("summarization", model="t5-small")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # the "request" must be passed into the template context
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process/")
async def process_pdf(file: UploadFile = File(...), template_id: str = Form(...)):

    save_dir = "temp-data-rec"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, file.filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    print(f"Saved upload to: {file_path}")

    extracted_text = await extract_text_from_handwritten()
    print(f"printing from /process route : {extracted_text}")
    #delete temp file.
    os.remove(file_path)
    # summary = await summarize_text(extracted_text)
    summary = summarizer(extracted_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    formatted_summary = format_summary(summary, template_id)
    print(f"Formatted summary: {formatted_summary}")

   
 # with open("output.txt", "w") as f:
    #     f.write(extracted_text)
    # with open("summary.txt", "w") as f:
    #     f.write(formatted_summary)
    return {
        "extracted_text": extracted_text,
        "summary": formatted_summary
    }

# @app.get("/download_text/")
# def download_text():
#     return FileResponse("output.txt")

# @app.get("/download_summary/")
# def download_summary():
#     return FileResponse("summary.txt")


def format_summary(summary, template_id):
    if template_id == "1":
        return f"Patient Visit Summary:\n\n{summary}"
    elif template_id == "2":
        return f"Medical Case Notes:\n\n{summary}"
    elif template_id == "3":
        return f"Historical Record Summary:\n\n{summary}"
    return summary



async def summarize_text(text):
    max_chunk = 500
    text = text.replace("\n", " ")
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ''
    for chunk in chunks:
        summary += summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] + ' '
    return summary.strip()

async def extract_text_from_handwritten():
    # Load the model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Directory with images
    image_folder = r"D:\RoboticsLab\text_summarizer\temp-data-rec"
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Loop over each image
    for image_file in image_files:
        try:
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB mode

            # Create multimodal chat prompt with the image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": "Extract the text from this image. Read it and rewrite the same paragraph exactly as shown."
                        }
                    ]
                }
            ]

            # Build prompt using chat template
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            # Preprocess inputs (text + image)
            inputs = processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )

            # Move inputs to the model's device (GPU or CPU)
            device = model.device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            # Generate output tokens
            output_ids = model.generate(**inputs, max_new_tokens=1024)

            # Slice off only the generated part (not the prompt)
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
            ]

            # Decode the output text
            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Print the extracted content
            # print(f"\n🖼️ Text extracted from {image_file}:\n{output_text[0]}")
            # print("-" * 60)
            return output_text[0]

        except Exception as e:
            print(f"\n❌ Error processing {image_file}: {e}")
            print("-" * 60)
            return e

# if __name__ == "__main__":
    # extract_text_from_handwritten()
