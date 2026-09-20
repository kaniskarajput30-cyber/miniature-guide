"""
Business Card -> Lead List extractor, powered by Qwen-VL (Vision-Language Model).
Streamlit version, for deployment on Streamlit Community Cloud.

- Users bulk-upload business card images.
- Each image is sent to a Qwen-VL model (via Alibaba Cloud DashScope's
  OpenAI-compatible API) with an instruction to return structured JSON.
- Results are shown in an editable table inside the app.
- Users can download the final lead list as an Excel file.

Run locally:
    pip install -r requirements.txt
    export DASHSCOPE_API_KEY="sk-..."   # https://dashscope.console.aliyun.com/apiKey
    streamlit run web_app.py

Deploy on Streamlit Community Cloud:
    1. Push web_app.py + requirements.txt to a GitHub repo.
    2. https://share.streamlit.io -> New app -> pick the repo -> main file: web_app.py
    3. In "Advanced settings -> Secrets", add:
         DASHSCOPE_API_KEY = "sk-..."
"""

import base64
import io
import json
import os
import re
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
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

def get_api_key(user_supplied: str) -> str:
    key = (user_supplied or "").strip()
    if key:
        return key
    # Prefer Streamlit secrets (used on Streamlit Cloud), fall back to env var.
    try:
        key = str(st.secrets.get("DASHSCOPE_API_KEY", "")).strip()
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    return key


def get_client(api_key: str, base_url: str) -> OpenAI:
    key = get_api_key(api_key)
    if not key:
        raise ValueError(
            "No API key found. Enter your DashScope / Qwen API key in the "
            "sidebar, or set DASHSCOPE_API_KEY in Streamlit secrets / env vars."
        )
    return OpenAI(api_key=key, base_url=(base_url or DEFAULT_BASE_URL).strip())

def image_to_data_uri(file_bytes: bytes, max_side: int = 1600) -> str:
    """Load image bytes, downscale if huge, and return a base64 data: URI."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
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
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model output: {text[:300]}")

def call_qwen_vl(client: OpenAI, model: str, file_bytes: bytes) -> dict:
    data_uri = image_to_data_uri(file_bytes)
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

def process_cards(uploaded_files, api_key, base_url, model):
    client = get_client(api_key, base_url)

    rows = []
    errors = []
    total = len(uploaded_files)
    progress = st.progress(0, text=f"Processing 0/{total}...")

    for i, uf in enumerate(uploaded_files):
        fname = uf.name
        progress.progress(i / total, text=f"Reading {fname} ({i + 1}/{total})")
        try:
            file_bytes = uf.getvalue()
            fields = call_qwen_vl(client, model.strip() or DEFAULT_MODEL, file_bytes)
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

    progress.progress(1.0, text="Done")
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df, errors

def build_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Leads")
        ws = writer.sheets["Leads"]
        for col_cells in ws.columns:
            length = max(
                (len(str(c.value)) if c.value is not None else 0) for c in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(
                max(length + 2, 12), 45
            )
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Business Card Lead Extractor (Qwen-VL)", layout="wide")

st.title("📇 Business Card → Lead List Extractor")
st.caption("Powered by **Qwen-VL** (vision-language model)")

if "leads_df" not in st.session_state:
    st.session_state.leads_df = pd.DataFrame(columns=COLUMNS)

with st.sidebar:
    st.header("⚙️ Model settings")
    st.markdown(
        "Uses Alibaba Cloud DashScope's OpenAI-compatible API for Qwen-VL. "
        "Get a free API key at "
        "[dashscope.console.aliyun.com/apiKey](https://dashscope.console.aliyun.com/apiKey)."
  )
    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-... (leave blank to use secrets/env var)",
    )
    base_url_input = st.text_input("API Base URL", value=DEFAULT_BASE_URL)
    model_input = st.text_input("Model", value=DEFAULT_MODEL)

st.subheader("1. Upload business card images (bulk)")
uploaded_files = st.file_uploader(
    "Drop images here",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    accept_multiple_files=True,
)
if uploaded_files:
    with st.expander(f"Preview ({len(uploaded_files)} image(s))", expanded=False):
        cols = st.columns(4)
        for i, uf in enumerate(uploaded_files):
            with cols[i % 4]:
                st.image(uf, caption=uf.name, use_container_width=True)

extract_clicked = st.button(
    "🔍 Extract Leads", type="primary", disabled=not uploaded_files
)

if extract_clicked:
    try:
        df, errors = process_cards(
            uploaded_files, api_key_input, base_url_input, model_input
        )
 st.session_state.leads_df = df
        ok = len(df) - len(errors)
        if errors:
            st.warning(
                f"Processed {len(df)} card(s): {ok} succeeded, {len(errors)} failed."
            )
            with st.expander("Show errors"):
                for e in errors:
                    st.write(f"- {e}")
        else:
            st.success(f"Processed {len(df)} card(s) successfully.")
    except ValueError as e:
        st.error(str(e))

st.subheader("2. Extracted leads (editable before export)")
edited_df = st.data_editor(
    st.session_state.leads_df,
    num_rows="dynamic",
    use_container_width=True,
    key="leads_editor",
)
st.session_state.leads_df = edited_df

st.subheader("3. Download")
if len(edited_df) > 0:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "⬇️ Download as Excel",
        data=build_excel(edited_df),
        file_name=f"leads_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.caption("Upload cards and click 'Extract Leads' to enable download.")
