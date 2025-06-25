import boto3
import os
import json
from dotenv import load_dotenv

load_dotenv()

BUCKET = "dpoimages"
REGION = os.getenv("AWS_REGION")
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=REGION
)

def generate_presigned_url(key):
    return s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=3600*24*7)  # 7 days

def build_image_sets(orig_prefix, gen_prefix, output_filename):
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=orig_prefix)
    image_sets = []

    for obj in response.get("Contents", []):
        orig_key = obj["Key"]
        if orig_key.endswith("/"): continue

        orig_filename = os.path.basename(orig_key)  # e.g., ART_Image_1.jpg
        base = os.path.splitext(orig_filename)[0]   # e.g., ART_Image_1
        orig_url = generate_presigned_url(orig_key)

        gen_urls = []
        for i in range(5):
            gen_key = f"{gen_prefix}generated_{base}_{i}.png"  # assuming gen images are .png
            try:
                gen_url = generate_presigned_url(gen_key)
                gen_urls.append(gen_url)
            except:
                print(f"⚠ Missing: {gen_key}")

        if len(gen_urls) == 5:
            image_sets.append({
                "original": orig_url,
                "generated": gen_urls
            })

    # Save JSON
    with open(output_filename, "w") as f:
        json.dump(image_sets, f, indent=2)

    print(f"✅ Written {len(image_sets)} image sets to {output_filename}")

# Call for both datasets
build_image_sets("bone_marrow_train_flat/", "bone_marrow_generated_flat/", "bone_marrow_image_sets.json")
build_image_sets("ham_10000_train_flat/", "genrated_images_flat/", "derma_image_sets.json")
