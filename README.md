# 🎯 DisCoRec: LLM-Guided Disentangled Conformity-aware Recommendation
Minkyung Song, Soyoung Park, Sungsu Lim*

## 🧩 Framework
<img width="1640" height="431" alt="WWW_framework" src="https://github.com/user-attachments/assets/0999a1fd-9a58-45e4-b120-df86d7961b68" />  

&nbsp;

**Model Specifications:**
- **LLM Generator:** Qwen2.5-14B-Instruct
- **Encoders:** text-embedding-ada-002, text-embedding-3-large

## 📦 Dependencies
### 1) Create Conda Environment
```bash
conda env create -f environment.yml
conda activate discorec_env
```
### 2) Install Additional Python Packages
```bash
pip install -r requirements.txt
```

## 📚 Dataset Structure and Download

**Amazon-book**/ **Amazon-movie**

You can download intent-based semantic embedding files in the following datasets:
Amazon-book/ Amazon-movie [GoogleDrive](https://drive.google.com/drive/folders/1rd2cppCrpoydvI1yvg5sIK2S68sBcn70?usp=sharing)

```plaintext
- amazon_book (/amazon_movie)
|--- trn_mat.pkl # training set (sparse matrix)
|--- val_mat.pkl # validation set (sparse matrix)
|--- tst_mat.pkl # test set (sparse matrix)
|--- usr_emb_np.pkl # user text embeddings
|--- itm_emb_np.pkl # item text embeddings
|--- user_intent_emb_3.pkl # user intent embeddings
|--- item_intent_emb_3.pkl # item intent embeddings
|--- user_conf_emb.pkl # user conformity embeddings
|--- item_conf_emb.pkl # item conformity embeddings
```
Amazon-Book: Uses the preprocessed split provided by RLMRec.
Amazon-Movie: Uses a reprocessed split prepared for this project.

## 🚀 Train & Evaluate

- **AlphaRec**
  ```bash
  python encoder/train_encoder.py --model alpharec --dataset {dataset} --cuda 0
  ```

- **LightGCN**
  ```bash
  python encoder/train_encoder.py --model lightgcn --dataset {dataset} --cuda 0
  ```

- **RLMRec**
  ```bash
  python encoder/train_encoder.py --model lightgcn_plus --dataset {dataset} --cuda 0
  ```
  ```bash
  python encoder/train_encoder.py --model lightgcn_gene --dataset {dataset} --cuda 0
  ```
  
- **IRLLRec**
  ```bash
  python encoder/train_encoder.py --model lightgcn_int --dataset {dataset} --cuda 0
  ```

- **DisCoRec (Ours)**
  ```bash
  python encoder/train_encoder.py --model discorec --dataset {dataset} --cuda 0
  ```

⚙️ Hyperparameters:

The hyperparameters of each model are stored in encoder/config/modelconf.

🙌 Acknowledgement

For fair comparison and reproducibility, we reuse parts of the IRLLRec and RLMRec codebases (training/evaluation routines and related utilities). We also adapt user/item profiling and embedding pipeline components. Source repositories:

> [RLMRec](https://github.com/HKUDS/RLMRec)
> 
> [IRLLRec](https://github.com/wangyu0627/IRLLRec)
>
Many thanks to them for providing the training framework and for the active contribution to the open source community.

