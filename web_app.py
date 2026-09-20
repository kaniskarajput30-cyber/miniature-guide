"""
Business Card -> Lead List extractor, powered by Qwen-VL (Vision-Language Model).

- Users bulk-upload business card images.
- Each image is sent to a Qwen-VL model (via Alibaba Cloud DashScope's
  OpenAI-compatible API) with an instruction to return structured JSON.
- Results are shown in an editable table inside the app.
- Users can download the final lead list as an .xlsx file.

Run:
    pip install -r requirements.txt
    export DASHSCOPE_API_KEY="sk-..." # get a free key at https://dashscope.console.aliyun.com/apiKey
    python app.py
"""

import base64
import io
import json
import os
import re
import traceback
from datetime import datetime

import gradio as gr
import pandas as pd
from openai import OpenAI
from PIL import Image

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

COLUMNS = [
    "First Name",
    "Last Name",
    "Position / Job Title",
    "Company",
    "Location",
    "Phone Number",
    "Email Address",
    "Source File",
]

FIELD_KEYS = [
    "first_name",
    "last_name",
    "position",
    "company",
    "location",
    "phone",
    "email",
]

# DashScope's OpenAI-compatible endpoint. Works with any OpenAI-compatible
# Qwen-VL deployment if you swap the base_url / model name.
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max"

SYSTEM_PROMPT = (
    "You are an information-extraction engine specialized in reading business "
    "cards. You always respond with a single valid JSON object and nothing else "
    "(no markdown fences, no commentary)."
)

USER_PROMPT = """Look at this business card image and extract the following fields.
If a field is not present on the card, return an empty string "" for it.
Do not guess or invent information that is not visible on the card.

Return ONLY a JSON object with exactly these keys:
{
  "first_name": "",
  "last_name": "",
  "position": "",
  "company": "",
  "location": "",
  "phone": "",
  "email": ""
}

Rules:
- "location" should be the city/region or address shown on the card (not the company name).
- "phone" should keep the digits and formatting as printed (include country code if shown).
- If the full name can't be cleanly split into first/last, put your best guess for first_name and the remainder in last_name.
- Output valid JSON only.
"""

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def get_client(api_key: str, base_url: str) -> OpenAI:
    key = (api_key or os.environ.get("DASHSCOPE_API_KEY", "")).strip()
    if not key:
        raise ValueError(
            "No API key provided. Enter your DashScope / Qwen API key in the "
            "'API Key' box, or set the DASHSCOPE_API_KEY environment variable."
        )
    return OpenAI(api_key=key, base_url=base_url.strip() or DEFAULT_BASE_URL)

def image_to_data_uri(path: str, max_side: int = 1600) -> str:
    """Load an image, downscale if huge, and return a base64 data: URI."""
    img = Image.open(path)
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from the model's reply."""
    text = text.strip()
    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the first {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model output: {text[:300]}")

def call_qwen_vl(client: OpenAI, model: str, image_path: str) -> dict:
    data_uri = image_to_data_uri(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    parsed = extract_json(raw)
    return {k: str(parsed.get(k, "") or "").strip() for k in FIELD_KEYS}

# ----------------------------------------------------------------------------
# Gradio callbacks
# ----------------------------------------------------------------------------

def process_cards(files, api_key, base_url, model, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload at least one business card image first.")

    try:
        client = get_client(api_key, base_url)
    except ValueError as e:
        raise gr.Error(str(e))

    rows = []
    errors = []
    total = len(files)

    for i, f in enumerate(files):
        path = f if isinstance(f, str) else f.name
        fname = os.path.basename(path)
        progress((i, total), desc=f"Reading {fname} ({i + 1}/{total})")
        try:
            fields = call_qwen_vl(client, model.strip() or DEFAULT_MODEL, path)
            rows.append(
                [
                    fields["first_name"],
                    fields["last_name"],
                    fields["position"],
                    fields["company"],
                    fields["location"],
                    fields["phone"],
                    fields["email"],
                    fname,
                ]
            )
        except Exception as e:
            traceback.print_exc()
            errors.append(f"{fname}: {e}")
            rows.append(["", "", "", "", "", "", "", f"{fname} (FAILED: {e})"])

    progress((total, total), desc="Done")
    df = pd.DataFrame(rows, columns=COLUMNS)

    status = f"Processed {total} card(s): {total - len(errors)} succeeded, {len(errors)} failed."
    if errors:
        status += "\n\nErrors:\n" + "\n".join(f"- {e}" for e in errors)

    return df, status

def export_excel(df: pd.DataFrame):
    if df is None or len(df) == 0:
        raise gr.Error("No leads to export yet. Upload cards and click 'Extract Leads' first.")

    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("outputs", f"leads_{ts}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Leads")
        ws = writer.sheets["Leads"]
        # Auto-width columns for readability.
        for col_cells in ws.columns:
            length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 12), 45)

    return out_path

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

with gr.Blocks(title="Business Card Lead Extractor (Qwen-VL)") as demo:
    gr.Markdown(
        """
        # 📇 Business Card → Lead List Extractor
        Powered by **Qwen-VL** (vision-language model).

        Upload business card images in bulk, extract structured contact fields,
        review/edit them, and download the result as an Excel file.
        """
    )

    with gr.Accordion("⚙️ Model settings", open=False):
        gr.Markdown(
            "Uses Alibaba Cloud DashScope's OpenAI-compatible API for Qwen-VL. "
            "Get a free API key at https://dashscope.console.aliyun.com/apiKey "
            "(or set the `DASHSCOPE_API_KEY` environment variable before launch)."
        )
        api_key_box = gr.Textbox(
            label="API Key",
            type="password",
            placeholder="sk-... (leave blank to use DASHSCOPE_API_KEY env var)",
        )
        model_box = gr.Textbox(label="Model", value=DEFAULT_MODEL)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload business card images (bulk)",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            gallery = gr.Gallery(label="Preview", columns=4, height=220)
            extract_btn = gr.Button("🔍 Extract Leads", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=3)
           with gr.Column(scale=2):
            leads_table = gr.Dataframe(
                headers=COLUMNS,
                datatype=["str"] * len(COLUMNS),
                row_count=(0, "dynamic"),
                col_count=(len(COLUMNS), "fixed"),
                interactive=True,
                label="Extracted Leads (editable before export)",
                wrap=True,
            )
            download_btn = gr.Button("⬇️ Download as Excel")
            download_file = gr.File(label="Your lead list", interactive=False)

    file_input.change(lambda files: files or [], inputs=file_input, outputs=gallery)

    extract_btn.click(
        process_cards,
        inputs=[file_input, api_key_box, base_url_box, model_box],
        outputs=[leads_table, status_box],
    )
download_btn.click(export_excel, inputs=leads_table, outputs=download_file)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
