import io
import os
import shutil
import zipfile
import pandas as pd
import requests
from tqdm import tqdm

RAW_CSV_PATH = "vit/test/LUNA25_Public_Training_Development_Data.csv"
MINI_CSV_PATH = "vit/test/mini_luna25_10.csv"
FINAL_IMG_DIR = "vit/test/mini_images_10"

N_BENIGN = 5      
N_MALIGNANT = 5     
RANDOM_SEED = 42
IMG_RECORD_ID = "14223624"
# ===============================================

os.makedirs(FINAL_IMG_DIR, exist_ok=True)


class RemoteSplitZipFile(io.RawIOBase):
    """將 Zenodo 上的 44 個分割壓縮檔虛擬串接成單一可隨機讀取的雲端檔案"""

    def __init__(self, splits_info):
        self.splits = []
        offset = 0
        for item in splits_info:
            size = item["size"]
            self.splits.append({
                "name": item["key"],
                "url": item["links"]["self"],
                "start": offset,
                "end": offset + size,
                "size": size,
            })
            offset += size
        self.total_size = offset
        self.pos = 0
        self.session = requests.Session()

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total_size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if self.pos >= self.total_size or size == 0:
            return b""
        if size < 0 or self.pos + size > self.total_size:
            size = self.total_size - self.pos

        chunks = []
        remaining = size
        while remaining > 0 and self.pos < self.total_size:
            part = next(s for s in self.splits if s["start"] <= self.pos < s["end"])
            local_start = self.pos - part["start"]
            read_len = min(remaining, part["end"] - self.pos)
            local_end = local_start + read_len - 1

            headers = {"Range": f"bytes={local_start}-{local_end}"}
            resp = self.session.get(part["url"], headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.content

            chunks.append(data)
            self.pos += len(data)
            remaining -= len(data)

        return b"".join(chunks)

    def readinto(self, b):
        """補齊 io.BufferedReader 所需的底層緩衝區寫入方法"""
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def main():
    # 1. 建立抽樣白名單
    print("1. 正在建立抽樣白名單...")
    dataset = pd.read_csv(RAW_CSV_PATH, dtype={"PatientID": str, "SeriesInstanceUID": str})

    malig_pool = dataset[dataset["label"] == 1]["PatientID"].drop_duplicates()
    malig_pts = malig_pool.sample(n=min(N_MALIGNANT, len(malig_pool)), random_state=RANDOM_SEED)

    benign_pool = dataset[(dataset["label"] == 0) & (~dataset["PatientID"].isin(malig_pts))]["PatientID"].drop_duplicates()
    benign_pts = benign_pool.sample(n=min(N_BENIGN, len(benign_pool)), random_state=RANDOM_SEED)

    target_pts = pd.concat([benign_pts, malig_pts])
    mini_df = dataset[dataset["PatientID"].isin(target_pts)].copy()
    target_uids = set(mini_df["SeriesInstanceUID"].unique())
    print(f"   已抽出 {len(target_pts)} 位病人，目標抓取 {len(target_uids)} 個 .mha 影像檔。")

    # 2. 取得 Zenodo 分割檔結構並開啟雲端虛擬 ZIP 目錄
    print("\n2. 正在連線 Zenodo 讀取遠端 ZIP 目錄表（僅需幾秒鐘）...")
    resp = requests.get(f"https://zenodo.org/api/records/{IMG_RECORD_ID}", timeout=30)
    resp.raise_for_status()
    splits = sorted(
        [f for f in resp.json().get("files", []) if f["key"].startswith("luna25_images.zip.")],
        key=lambda x: x["key"],
    )

    remote_zip_stream = RemoteSplitZipFile(splits)
    buffered_stream = io.BufferedReader(remote_zip_stream, buffer_size=1024 * 1024)

    with zipfile.ZipFile(buffered_stream, "r") as zf:
        all_zip_entries = zf.namelist()
        print(f"   成功讀取遠端目錄！壓縮包內共有 {len(all_zip_entries)} 個檔案。")

        matched_entries = [
            entry for entry in all_zip_entries
            if entry.endswith(".mha") and os.path.splitext(os.path.basename(entry))[0] in target_uids
        ]
        print(f"   成功定位到 {len(matched_entries)} 個目標影像，開始精準串流下載...\n")

        # 3. 逐一從雲端壓縮包抽出目標 .mha 並存到本地
        for entry in tqdm(matched_entries, desc="下載並解壓縮進度"):
            filename = os.path.basename(entry)
            dest_path = os.path.join(FINAL_IMG_DIR, filename)
            if os.path.exists(dest_path):
                continue

            with zf.open(entry) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)

    # 4. 同步輸出對應的精簡 CSV
    downloaded_uids = {f.replace(".mha", "") for f in os.listdir(FINAL_IMG_DIR) if f.endswith(".mha")}
    verified_df = mini_df[mini_df["SeriesInstanceUID"].isin(downloaded_uids)]
    verified_df.to_csv(MINI_CSV_PATH, index=False)

    print(f"\n完成！實際下載 {len(downloaded_uids)} 個未處理原始 .mha 影像檔。")
    print(f"影像存放於：{FINAL_IMG_DIR}")
    print(f"標註檔儲存於：{MINI_CSV_PATH}")


if __name__ == "__main__":
    main()