"""
Business Card Lead Extractor
=============================

Streamlit app that lets a user upload multiple business card images in bulk,
sends each image to a Qwen Vision-Language Model (Qwen-VL, via Alibaba
Cloud's DashScope API) to extract structured contact fields, displays the
resulting lead list in the app, and lets the user download it as an Excel
(.xlsx) file.

Run locally:
    pip install -r requirements.txt
    export DASHSCOPE_API_KEY="sk-..."          # or put it in .streamlit/secrets.toml
    streamlit run app.py

Deploy: see README.md for GitHub + Streamlit Community Cloud steps.
"""

import base64
import io
import json
import os
import time
from dataclasses import dataclass, fields
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:  # dependency not installed yet — surfaced nicely in the UI
    dashscope = None
    MultiModalConversation = None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

APP_TITLE = "Business Card Lead Extractor"
# Qwen-VL model served via DashScope. "qwen-vl-plus" is fast/cheap and works
# well for OCR-style extraction; swap to "qwen-vl-max" for higher accuracy.
DEFAULT_MODEL = "qwen-vl-plus"
MAX_IMAGE_EDGE = 1600  # downscale very large photos before sending, saves tokens/time
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]

LEAD_FIELDS = [
    "First Name",
    "Last Name",
    "Position / Job Title",
    "Company",
    "Location",
    "Phone Number",
    "Email Address",
]

EXTRACTION_PROMPT = """You are an information-extraction assistant reading a photo of a single \
business card.

Extract the following fields from the card and return ONLY a single valid JSON object \
(no markdown fences, no commentary) with exactly these keys:

- "first_name"
- "last_name"
- "job_title"
- "company"
- "location"
- "phone"
- "email"

Rules:
- If a field is not present on the card, use an empty string "" for it.
- Split the person's full name into first_name and last_name as best you can.
- "location" means the office address / city / country printed on the card \
(keep it concise — city, state/region, country if available). Do not include the \
company name here.
- If multiple phone numbers are printed, pick the primary/mobile one; if unsure, \
use the first one listed.
- If multiple emails are printed, use the first one.
- Do not invent information that is not on the card.
- Return valid JSON and nothing else.
"""


@dataclass
class Lead:
    first_name: str = ""
    last_name: str = ""
    job_title: str = ""
    company: str = ""
    location: str = ""
    phone: str = ""
    email: str = ""
    source_file: str = ""
    status: str = "ok"  # "ok" or "error"
    error_message: str = ""

    def to_display_row(self) -> dict:
        return {
            "First Name": self.first_name,
            "Last Name": self.last_name,
            "Position / Job Title": self.job_title,
            "Company": self.company,
            "Location": self.location,
            "Phone Number": self.phone,
            "Email Address": self.email,
            "Source File": self.source_file,
            "Status": self.status,
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """Look for the DashScope API key in Streamlit secrets, then env vars."""
    try:
        if "DASHSCOPE_API_KEY" in st.secrets:
            return st.secrets["DASHSCOPE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("DASHSCOPE_API_KEY")


def downscale_image(image_bytes: bytes) -> bytes:
    """Downscale very large images to keep API calls fast/cheap. Returns JPEG bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > MAX_IMAGE_EDGE:
            scale = MAX_IMAGE_EDGE / float(longest)
            img = img.resize((int(w * scale), int(h * scale)))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=90)
        return out.getvalue()
    except Exception:
        # If Pillow can't process it for some reason, fall back to original bytes.
        return image_bytes


def image_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def extract_json_from_text(text: str) -> dict:
    """Qwen-VL sometimes wraps JSON in markdown fences or adds stray text.
    Pull out the first {...} block and parse it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model response: {text[:200]}")
    candidate = text[start : end + 1]
    return json.loads(candidate)


def call_qwen_vl(image_bytes: bytes, api_key: str, model: str) -> dict:
    """Send one business card image to Qwen-VL via DashScope and return the parsed fields."""
    dashscope.api_key = api_key
    data_url = image_to_data_url(image_bytes)

    messages = [
        {
            "role": "user",
            "content": [
                {"image": data_url},
                {"text": EXTRACTION_PROMPT},
            ],
        }
    ]

    response = MultiModalConversation.call(model=model, messages=messages)

    if response.status_code != 200:
        raise RuntimeError(
            f"DashScope API error {response.status_code}: "
            f"{getattr(response, 'message', 'unknown error')}"
        )

    # Response content is a list of segments; extract the text segment.
    content = response.output.choices[0].message.content
    if isinstance(content, list):
        text = "".join(seg.get("text", "") for seg in content if isinstance(seg, dict))
    else:
        text = str(content)

    return extract_json_from_text(text)


def process_card(
    file_name: str, image_bytes: bytes, api_key: str, model: str, retries: int = 2
) -> Lead:
    small_bytes = downscale_image(image_bytes)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            parsed = call_qwen_vl(small_bytes, api_key, model)
            return Lead(
                first_name=str(parsed.get("first_name", "")).strip(),
                last_name=str(parsed.get("last_name", "")).strip(),
                job_title=str(parsed.get("job_title", "")).strip(),
                company=str(parsed.get("company", "")).strip(),
                location=str(parsed.get("location", "")).strip(),
                phone=str(parsed.get("phone", "")).strip(),
                email=str(parsed.get("email", "")).strip(),
                source_file=file_name,
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001 — surface any failure per-card
            last_error = str(exc)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            continue
    return Lead(source_file=file_name, status="error", error_message=last_error)


def leads_to_dataframe(leads: list) -> pd.DataFrame:
    rows = [lead.to_display_row() for lead in leads]
    return pd.DataFrame(rows, columns=[
        "First Name",
        "Last Name",
        "Position / Job Title",
        "Company",
        "Location",
        "Phone Number",
        "Email Address",
        "Source File",
        "Status",
    ])


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Leads")
        worksheet = writer.sheets["Leads"]
        # Auto-size columns a bit for readability.
        for i, col in enumerate(df.columns, start=1):
            max_len = max(
                [len(str(col))] + [len(str(v)) for v in df[col].astype(str).tolist()]
            )
            worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = min(
                max(12, max_len + 2), 45
            )
    return output.getvalue()


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📇", layout="wide")

    st.title("📇 Business Card Lead Extractor")
    st.caption(
        "Upload a batch of business card photos and let a Qwen Vision-Language "
        "Model turn them into a clean, downloadable lead list."
    )

    if dashscope is None:
        st.error(
            "The `dashscope` package isn't installed. Run "
            "`pip install -r requirements.txt` and restart the app."
        )
        st.stop()

    # ---- Sidebar: settings ----
    with st.sidebar:
        st.header("Settings")
        api_key = get_api_key()
        api_key_input = st.text_input(
            "DashScope API key",
            value=api_key or "",
            type="password",
            help=(
                "Get a key from https://dashscope.console.aliyun.com/apiKey. "
                "For deployed apps, set this as a Streamlit secret named "
                "DASHSCOPE_API_KEY instead of typing it here."
            ),
        )
        model_choice = st.selectbox(
            "Qwen-VL model",
            options=["qwen-vl-plus", "qwen-vl-max", "qwen-vl-ocr"],
            index=0,
            help=(
                "qwen-vl-plus: fast and cheap, good default.\n"
                "qwen-vl-max: highest accuracy, slower/costlier.\n"
                "qwen-vl-ocr: tuned for dense text extraction."
            ),
        )
        st.markdown("---")
        st.caption(
            "Images never leave this session except to be sent to the Qwen-VL "
            "API for extraction."
        )

    effective_key = api_key_input or api_key

    # ---- Main: upload ----
    uploaded_files = st.file_uploader(
        "Upload business card images (you can select multiple files at once)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} image(s) selected.**")
        with st.expander("Preview uploaded images", expanded=False):
            cols = st.columns(min(4, len(uploaded_files)))
            for i, f in enumerate(uploaded_files):
                with cols[i % len(cols)]:
                    st.image(f, caption=f.name, use_container_width=True)
                f.seek(0)  # reset pointer after preview read

    run_disabled = not uploaded_files or not effective_key
    if uploaded_files and not effective_key:
        st.warning("Enter your DashScope API key in the sidebar to run extraction.")

    if st.button("🚀 Extract Leads", type="primary", disabled=run_disabled):
        leads = []
        progress = st.progress(0.0, text="Starting...")
        status_area = st.empty()

        total = len(uploaded_files)
        for idx, file in enumerate(uploaded_files):
            status_area.info(f"Processing **{file.name}** ({idx + 1}/{total})...")
            file.seek(0)
            image_bytes = file.read()
            lead = process_card(file.name, image_bytes, effective_key, model_choice)
            leads.append(lead)
            progress.progress((idx + 1) / total, text=f"Processed {idx + 1}/{total}")

        status_area.empty()
        progress.empty()

        st.session_state["leads_df"] = leads_to_dataframe(leads)

        n_ok = sum(1 for l in leads if l.status == "ok")
        n_err = len(leads) - n_ok
        if n_err:
            st.warning(f"Extracted {n_ok} card(s) successfully, {n_err} failed. See table below.")
        else:
            st.success(f"Extracted all {n_ok} card(s) successfully!")

    # ---- Results ----
    if "leads_df" in st.session_state:
        df = st.session_state["leads_df"]
        st.subheader("Extracted Leads")
        st.dataframe(df, use_container_width=True, hide_index=True)

        excel_bytes = dataframe_to_excel_bytes(df)
        st.download_button(
            label="⬇️ Download Lead List (Excel)",
            data=excel_bytes,
            file_name="business_card_leads.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
