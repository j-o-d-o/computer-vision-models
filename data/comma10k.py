import cv2
from pymongo import MongoClient
import argparse
from tqdm import tqdm
from data.label_spec import Entry


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Upload semseg data from comma10k dataset")
  parser.add_argument("--src_path", type=str, help="Path to comma10k dataset e.g. /home/user/comma10k")
  parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
  parser.add_argument("--db", type=str, default="semseg", help="MongoDB database")
  parser.add_argument("--collection", type=str, default="comma10k", help="MongoDB collection")
  args = parser.parse_args()

  client = MongoClient(args.conn)
  collection = client[args.db][args.collection]

  with open(args.src_path + "/files_trainable") as f:
    for trainable_img in tqdm(f.readlines()):
      trainable_img = trainable_img.strip()
      name = trainable_img[6:]
      mask_path = args.src_path + "/" + trainable_img
      img_path = args.src_path + "/imgs/" + name

      if collection.count_documents({'name': name}, limit=1) == 0:
        mask_data = cv2.imread(mask_path)
        mask_bytes = cv2.imencode('.png', mask_data)[1].tobytes()
        img_data = cv2.imread(img_path)
        img_bytes = cv2.imencode('.png', img_data)[1].tobytes()
        entry = Entry(
          img=img_bytes,
          mask=mask_bytes,
          content_type="image/png",
          org_source="comma10k",
          org_id=name,
        )
        collection.insert_one(entry.get_dict())
      else:
        print("WARNING: " + name + " already exist, continue with next image")
