import streamlit as st
import sys
import json
import os
import re
import pandas as pd
import requests
from io import BytesIO
from io import StringIO
from datetime import datetime
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from streamlit_image_zoom import image_zoom
import ssl
import urllib3

try:
    from PIL import Image
except ImportError:
    st.error("Pillow library is not installed. Please install it using: pip install Pillow")
    Image = None

sys.path.append("project")
load_dotenv()

# Configure urllib3 to use TLS 1.2
urllib3.util.ssl_.DEFAULT_CIPHERS = 'DEFAULT@SECLEVEL=2'

# Create a requests session with retries
session = requests.Session()
session.mount('https://', requests.adapters.HTTPAdapter(
    max_retries=3,
    pool_connections=10,
    pool_maxsize=10
))

# -------------- Load Class Mappings ----------------
def load_class_mappings():
    """Load class name mappings from JSON file."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "class_mappings.json")
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Class mappings JSON file not found.")
        st.error(f"🔍 Tried: {path}")
        return {}
    except Exception as e:
        st.error(f"Error loading class mappings: {e}")
        return {}

CLASS_MAPPINGS = load_class_mappings()

def get_class_name(url):
    """Extract class short form from URL and map to full form."""
    try:
        match = re.search(r"/[^/]+/([^_]+)_image_", url)
        if match:
            short_form = match.group(1)
            return CLASS_MAPPINGS.get(short_form, "Unknown")
        return "Unknown"
    except Exception as e:
        st.error(f"Error extracting class name from URL: {e}")
        return "Unknown"

# -------------- URL Key Extraction Helper ----------------
def extract_key(url: str) -> str:
    """Extract the stable S3 key path fragment from a (possibly presigned) URL."""
    try:
        match = re.search(r"\.com/(.+?)(?:\?|$)", url)
        return match.group(1) if match else url
    except Exception:
        return url

# -------------- AWS + S3 Config ----------------
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    BUCKET_NAME = os.getenv("S3_BUCKET")
except Exception as e:
    st.error(f"Failed to initialize S3 client: {e}")
    BUCKET_NAME = None

def test_s3_connection():
    """Test S3 connectivity silently, only display errors."""
    if BUCKET_NAME is None:
        st.error("S3 bucket not configured.")
        return
    try:
        s3.list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=1)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        st.error(f"S3 connection failed: {error_code} - {e.response['Error']['Message']}")
    except Exception as e:
        st.error(f"S3 connection failed: {e}")

def upload_csv_to_s3(local_path, s3_key):
    if BUCKET_NAME is None:
        st.error("S3 bucket not configured.")
        return
    try:
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        print(f"✅ Uploaded {s3_key} to S3")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        st.error(f"Failed to upload CSV to S3: {error_code} - {e.response['Error']['Message']}")
    except Exception as e:
        st.error(f"Failed to upload CSV to S3: {e}")

def download_csv_from_s3(s3_key):
    """Download CSV from S3 and return as a pandas DataFrame."""
    if BUCKET_NAME is None:
        st.error("S3 bucket not configured.")
        return pd.DataFrame(columns=["Original_Image", "Generated_Image", "Plausibility", "Date"])
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        return df
    except s3.exceptions.NoSuchKey:
        return pd.DataFrame(columns=["Original_Image", "Generated_Image", "Plausibility", "Date"])
    except ClientError as e:
        error_code = e.response['Error']['Code']
        st.error(f"Failed to download CSV from S3: {error_code} - {e.response['Error']['Message']}")
        return pd.DataFrame(columns=["Original_Image", "Generated_Image", "Plausibility", "Date"])
    except Exception as e:
        st.error(f"Failed to download CSV from S3: {e}")
        return pd.DataFrame(columns=["Original_Image", "Generated_Image", "Plausibility", "Date"])

# Prefer local CSV (fresh writes) and fall back to S3
def get_annotations_df():
    """Load annotations, preferring local CSV for immediate freshness, else S3."""
    try:
        if os.path.exists(CSV_FILE):
            return pd.read_csv(CSV_FILE)
    except Exception as e:
        st.error(f"Failed reading local CSV: {e}")
    # Fallback to S3
    return download_csv_from_s3(S3_CSV_KEY)

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Image Annotation Tool")

# ---------------- Session State Setup ----------------
if "selected_task" not in st.session_state:
    from interface import app_selector
    app_selector()
    st.stop()
task = st.session_state.selected_task
CSV_FILE = f"project/{task}_annotations.csv"
S3_CSV_KEY = f"annotations/project/{task}_annotations.csv"

if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "selections" not in st.session_state:
    st.session_state.selections = {}
if "completed_sets" not in st.session_state:
    st.session_state.completed_sets = set()

# ---------------- Load URLs ----------------
from interface import get_image_sets
image_sets = get_image_sets()
if not image_sets:
    st.warning("No images available.")
    st.stop()
# Validate and reset current_index if out of bounds
if st.session_state.current_index >= len(image_sets):
    st.session_state.current_index = max(0, len(image_sets) - 1)

# ---------------- Image Fetching ----------------
def load_image(url):
    if Image is None:
        st.error("Cannot load images: Pillow library is missing.")
        return None
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except requests.exceptions.SSLError as e:
        st.error(f"Failed to load image {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load image {url}: {e}")
        return None
    except NameError as e:
        st.error(f"Image processing error: {e}. Ensure Pillow is installed.")
        return None
    except Exception as e:
        st.error(f"Error processing image {url}: {e}")
        return None

# ---------------- CSV Saving ----------------
def save_selection_to_csv(original_url, generated_url, label):
    original = extract_key(original_url)
    generated = extract_key(generated_url)

    data = {
        "Original_Image": original,
        "Generated_Image": generated,
        "Plausibility": label,
        "Date": datetime.now().strftime("%d-%m-%Y")
    }

    # Ensure project directory exists
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    # Load existing annotations (local preferred for freshness)
    df = get_annotations_df()

    # Append new data, avoiding duplicates
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True).drop_duplicates(
        subset=["Original_Image", "Generated_Image"], keep="last"
    )

    # Save locally and upload to S3
    df.to_csv(CSV_FILE, index=False)
    upload_csv_to_s3(CSV_FILE, S3_CSV_KEY)
    # Do not rerun here; only advance on explicit Next click

# ---------------- Counter for Fully Annotated Originals ----------------
def get_total_fully_annotated():
    """Count how many original images are fully annotated based on actual generated counts per set."""
    df = get_annotations_df()
    if df.empty:
        return 0
    try:
        # Build expected counts per original from current image_sets
        expected = {}
        for s in image_sets:
            orig_key = extract_key(s["original"])
            expected[orig_key] = len([extract_key(u) for u in s.get("generated", [])])

        fully_done = 0
        for orig_key, needed in expected.items():
            if needed == 0:
                continue
            done = df[df["Original_Image"] == orig_key]["Generated_Image"].nunique()
            if done >= needed:
                fully_done += 1
        return fully_done
    except Exception as e:
        st.error(f"Error counting fully annotated images: {e}")
        return 0

# ---------------- Navigation ----------------
def show_navigation():
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.session_state.current_index == 0:
            if st.button("Back to Task Selector"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            if st.button("Previous"):
                st.session_state.current_index -= 1
                if st.session_state.current_index < 0:
                    st.session_state.current_index = 0
                st.rerun()
    with col3:
        if st.button("Next"):
            index = st.session_state.current_index
            current_set = image_sets[index]

            original_key = extract_key(current_set["original"])
            df = get_annotations_df()
            annotations = df[df["Original_Image"] == original_key]
            generated_keys = [extract_key(url) for url in current_set["generated"]]

            # Include any selections made in this session (not relying solely on CSV)
            annotated_keys = set(annotations["Generated_Image"].astype(str).tolist())
            for (sel_index, i), _label in st.session_state.selections.items():
                if sel_index == index and i < len(current_set["generated"]):
                    annotated_keys.add(extract_key(current_set["generated"][i]))

            if len(set(generated_keys)) == len(annotated_keys.intersection(set(generated_keys))):
                st.session_state.completed_sets.add(index)
                if st.session_state.current_index < len(image_sets) - 1:
                    st.session_state.current_index += 1
                else:
                    st.session_state.current_index = 0  # optional: loop back
                st.rerun()   # 👈 force refresh here
            else:
                st.warning("⚠ Please annotate all generated images before proceeding.")


def is_fully_annotated(original_url, generated_urls, df):
    original_key = extract_key(original_url)
    annotations = df[df["Original_Image"] == original_key]
    generated_keys = [extract_key(url) for url in generated_urls]
    return annotations["Generated_Image"].nunique() == len(generated_keys)


# ---------------- Render Generated Image Fragment ----------------
@st.fragment
def render_generated_image(index, gen_url, i, original_url, annotations_df):
    fragment_key = f"gen_image_{index}_{i}"

    with st.container(key=fragment_key):
        cols = st.columns([2, 3])
        with cols[0]:
            gen_img = load_image(gen_url)
            if gen_img:
                gw, gh = gen_img.size
                gh_scaled = int(gh * 300 / gw)
                image_zoom(gen_img, size=(300, gh_scaled), zoom_factor=2.5)
            else:
                st.warning(f"Could not load generated image {i+1}")

        with cols[1]:
            key = (index, i)
            gen_key = extract_key(gen_url)
            
            if key not in st.session_state.selections:
                existing_annotation = annotations_df[annotations_df["Generated_Image"] == gen_key]
                if not existing_annotation.empty:
                    st.session_state.selections[key] = existing_annotation.iloc[0]["Plausibility"]

            selected = st.session_state.selections.get(key)

            margin_top = 60
            st.write(f"<div style='height: {margin_top}px;'></div>", unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])
            with c1:
                label = "✔ Plausible" if selected == "Plausible" else "Plausible"
                plausible_key = f"p_{index}_{i}_{gen_key[:10]}"
                
                if st.button(label, key=plausible_key):
                    st.session_state.selections[key] = "Plausible"
                    save_selection_to_csv(original_url, gen_url, "Plausible")
            with c2:
                label = "✔ Implausible" if selected == "Implausible" else "Implausible"
                implausible_key = f"ip_{index}_{i}_{gen_key[:10]}"
                if st.button(label, key=implausible_key):
                    st.session_state.selections[key] = "Implausible"
                    save_selection_to_csv(original_url, gen_url, "Implausible")

# ---------------- Main View ----------------
def show_main_view():
    df = get_annotations_df()

    # Skip to next unannotated image
    while st.session_state.current_index < len(image_sets):
        current_set = image_sets[st.session_state.current_index]
        if not is_fully_annotated(current_set["original"], current_set["generated"], df):
            break  # Found next unannotated
        st.session_state.current_index += 1

    # If all are annotated
    if st.session_state.current_index >= len(image_sets):
        st.success("🎉 All images have been annotated!")
        st.stop()

    index = st.session_state.current_index
    current_set = image_sets[index]

    original_key = extract_key(current_set["original"])
    df = get_annotations_df()

    st.markdown("### Original Image")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        original_img = load_image(current_set["original"])
        if original_img:
            ow, oh = original_img.size
            oh_scaled = int(oh * 400 / ow)
            image_zoom(original_img, size=(400, oh_scaled), zoom_factor=2.5)
            class_name = get_class_name(current_set["original"])
            st.markdown(f"**Class: {class_name}**")
        else:
            st.error(f"Could not load original image: {current_set['original']}")

    st.markdown("---")
    st.markdown("### Generated Images")
    for i, gen_url in enumerate(current_set["generated"]):
        render_generated_image(index, gen_url, i, current_set["original"], df)

    show_navigation()

# ---------------- Run ----------------
st.title("Image Annotation Tool")
st.success(f"Annotated Images: {get_total_fully_annotated()}")
st.success(f"Progress: {get_total_fully_annotated()} / {len(image_sets)} sets annotated.")


# Test S3 connection silently at startup
test_s3_connection()

show_main_view()
