# Business Card Lead Extractor

A Streamlit app that lets you upload a batch of business card photos, extracts
structured contact fields using a **Qwen Vision-Language Model** (Qwen-VL via
Alibaba Cloud's DashScope API), shows the results in a table, and lets you
download them as an Excel file.

Extracted fields:
- First Name
- Last Name
- Position / Job Title
- Company
- Location
- Phone Number
- Email Address

---

## 1. Get a DashScope API key

The app calls the hosted Qwen-VL model through Alibaba Cloud's DashScope API
(no GPU or local model download needed).

1. Go to https://dashscope.console.aliyun.com/apiKey (sign up / sign in — an
   Alibaba Cloud account is required; new accounts get some free quota).
2. Create an API key and copy it (starts with `sk-...`).
3. Keep it secret — treat it like a password.

---

## 2. Run it locally first (recommended)

```bash
git clone <your-repo-url>
cd biz-card-extractor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Option A: environment variable
export DASHSCOPE_API_KEY="sk-your-key-here"     # Windows (PowerShell): $env:DASHSCOPE_API_KEY="sk-..."

# Option B: local secrets file (not committed to git)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your real key in

streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), upload a few
business card images, click **Extract Leads**, and download the Excel file.

---

## 3. Push the project to GitHub

1. Create a new **empty** repository on GitHub (no README/license, so it
   doesn't conflict with the files here) — e.g. `business-card-lead-extractor`.
2. From inside this project folder:

```bash
git init
git add app.py requirements.txt README.md .gitignore .streamlit/config.toml .streamlit/secrets.toml.example
git commit -m "Add Qwen-VL business card lead extractor Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> **Important:** `.streamlit/secrets.toml` (with your real key) is listed in
> `.gitignore` and must never be committed. Only `secrets.toml.example`
> (a template with a placeholder) goes to GitHub.

---

## 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **"Create app"** → **"From an existing repo"**.
3. Pick your repository, the `main` branch, and set the main file path to
   `app.py`.
4. Before (or right after) deploying, open **Advanced settings → Secrets**
   and paste:

   ```toml
   DASHSCOPE_API_KEY = "sk-your-real-key-here"
   ```

   This keeps the key out of your GitHub repo entirely, and the app reads it
   automatically via `st.secrets`.
5. Click **Deploy**. After the build finishes (a minute or two), Streamlit
   gives you a public URL like `https://<your-app>.streamlit.app` — that's
   your live, shareable app.

Whenever you `git push` new changes to `main`, Streamlit Community Cloud
automatically redeploys.

### Alternative: deploy on Hugging Face Spaces
If you'd rather host on Hugging Face Spaces instead:
1. Create a new Space → SDK: **Streamlit**.
2. Push this same repo's files to the Space's git remote (`app.py`,
   `requirements.txt`, etc.).
3. In the Space's **Settings → Repository secrets**, add
   `DASHSCOPE_API_KEY` with your key.
4. The Space will build and serve the app automatically.

---

## How it works

1. **Upload** — `st.file_uploader(accept_multiple_files=True)` lets the user
   select many card images at once.
2. **Preprocess** — each image is downscaled (max edge ~1600px) and
   re-encoded as JPEG to keep API calls fast and cheap.
3. **Extract** — each image is sent to Qwen-VL (`qwen-vl-plus` by default,
   `qwen-vl-max` or `qwen-vl-ocr` selectable in the sidebar) with a prompt
   instructing it to return a strict JSON object with the seven target
   fields. The app parses that JSON per card, with a couple of automatic
   retries on transient failures.
4. **Display** — results are collected into a pandas DataFrame and shown in
   an interactive table, including a `Status` column flagging any card that
   failed extraction so you can retry it or check the image quality.
5. **Export** — the DataFrame is written to an in-memory `.xlsx` workbook
   (via `openpyxl`) with auto-sized columns, offered through
   `st.download_button` — no temp files touch disk on the server.

## Notes & tips

- **Accuracy**: `qwen-vl-max` generally reads dense or low-quality card
  photos more reliably than `qwen-vl-plus`; `qwen-vl-ocr` is worth trying for
  cards with dense small print.
- **Cost**: DashScope bills per call/token. Processing is done one image per
  API call, so cost scales linearly with the number of cards.
- **Privacy**: card images are sent to Alibaba Cloud's DashScope API for
  processing and are not stored by this app itself.
- **Rate limits**: for very large batches (100+ cards), consider adding a
  short delay between calls if you hit rate limits on your DashScope plan.
