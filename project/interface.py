# interface.py
import streamlit as st
import json
import os
import boto3
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import re

load_dotenv()

# -------------- AWS S3 Config ----------------
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
BUCKET_NAME = os.getenv("S3_BUCKET")

def download_csv_from_s3(s3_key):
    """Download CSV from S3 and return as a pandas DataFrame."""
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        return df
    except s3.exceptions.NoSuchKey:
        return pd.DataFrame(columns=["Original_Image", "Generated_Image", "Plausibility", "Date"])
    except Exception as e:
        print(f"⚠ Failed to download CSV from S3: {e}")
        return pd.DataFrame(columns=["Original_Image", "Generated_Image", "Plausibility", "Date"])

# ------------------- Task Selector UI -------------------
def app_selector():
    st.title("Select Annotation Task")
    st.markdown("Please choose the dataset you want to annotate:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Bone Marrow"):
            st.session_state.selected_task = "bone"
    with col2:
        if st.button("Dermatology"):
            st.session_state.selected_task = "derma"

# ------------------- Load JSON-based Image Sets -------------------
@st.cache_data
def load_image_sets_from_json(task):
    """Load image sets from JSON and filter out fully annotated images."""
    if task == "bone":
        json_path = "bone_marrow_image_sets.json"
        csv_key = "annotations/DPO/bone_annotations.csv"
    elif task == "derma":
        json_path = "derma_image_sets.json"
        csv_key = "annotations/DPO/derma_annotations.csv"
    else:
        return []

    if not os.path.exists(json_path):
        st.error(f"⚠ JSON file not found: {json_path}")
        return []

    try:
        with open(json_path, "r") as f:
            image_sets = json.load(f)
    except Exception as e:
        st.error(f"⚠ Failed to load image sets: {e}")
        return []

    # Download and read the CSV from S3
    df = download_csv_from_s3(csv_key)

    def extract_key(url):
        match = re.search(r"\.com/(.+?)(?:\?|$)", url)
        return match.group(1) if match else url

    # Filter out image sets where all generated images are annotated
    filtered_image_sets = []
    for image_set in image_sets:
        original_key = extract_key(image_set["original"])
        generated_keys = [extract_key(url) for url in image_set["generated"]]

        # Check annotations for this original image
        annotations = df[df["Original_Image"] == original_key]
        annotated_gen_keys = set(annotations["Generated_Image"].values)

        # If not all generated images are annotated, keep this set
        if len(annotated_gen_keys.intersection(set(generated_keys))) < len(generated_keys):
            filtered_image_sets.append(image_set)

    if not filtered_image_sets:
        st.warning("All images have been annotated for this task.")
        return []

    return filtered_image_sets

# ------------------- Wrapper Function -------------------
def get_image_sets():
    """Get image sets for the selected task."""
    task = st.session_state.get("selected_task")
    return load_image_sets_from_json(task)