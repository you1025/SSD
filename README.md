# SSD おためし

Single Shot Multibox Detector by PyTorch

# 実行手順

## データの準備

Pascal VOC 2012 データを用いる

```python
# リポジトリをクローン
git clone https://github.com/you1025/SSD.git

# 訓練データのダウンロード
cd SSD/data
sh setup_voc_data.sh
```

## モデルの学習

### 学習用設定ファイルの修正

`configs/train_config.yaml` を適宜修正する。  
基本的には下記が対象となるはず。

- data_loader.batch_size: デフォルトで 32
- train.epochs: デフォルトで 300

### 学習の実行

SSD ディレクトリの直下で下記を実行する。

```python
python ssd_train.py
```

## 学習モデルを用いた推論

### 推論用設定ファイルの修正

`configs/inference_config.yaml` を適宜修正する。  
基本的に修正は不要なはず。

### 推論用の画像の配置

`images/input/` ディレクトリに推論に用いたい画像を配置する。

### 推論の実行

SSD ディレクトリの直下で下記を実行する。

```python
python ssd_inference.py
```

処理が終了すると `images/output/` ディレクトリに推論済の画像が出力される。


# 改修したいこと

- 評価用の機能を追加(mAP etc.)
- 物体未検出の画像も出力対象とする
- VOC 2012 以外のデータも追加したい
  - VOC 2007
  - moco
- モデルの強化(試してみたい)
  - SSD512
  - DSSD: [物体検知と真剣に向き合ってみる (Prediction Module + Deconvolution編)](https://www.sigfoss.com/developer_blog/detail?actual_object_id=253) 参考
  - FSSD: [物体検知と真剣に向き合ってみる (FSSD編)](https://www.sigfoss.com/developer_blog/detail?actual_object_id=256) 参考


# 参考

- [つくりながら学ぶ！PyTorchによる発展ディープラーニング](https://book.mynavi.jp/ec/products/detail/id=104855)
- [PyTorchによる物体検出](https://www.amazon.co.jp/dp/B08J3RWCYZ/)
- [SSD:Single Shot Multibox Detector](https://qiita.com/de0ta/items/1ae60878c0e177fc7a3a)
