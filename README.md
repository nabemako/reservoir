# Reservoir Computing Implementation

このリポジトリは、**Echo State Networks (ESN)** を使用したリザーバーコンピューティングの実装です。

## 概要

リザーバーコンピューティングは、時系列予測などのタスクに適した機械学習手法です。

### アーキテクチャ

```
入力層 → [リザーバー層] → 出力層（訓練）
         固定ランダム重み
```

- **入力層**: ランダムに初期化された重み行列 `W_in`
- **リザーバー層**: 固定のスパース（疎）な重み行列 `W_res`
- **出力層**: Ridge回帰で訓練された重み `W_out`

## インストール

### 必要なパッケージ

```bash
pip install -r requirements.txt
```

または個別にインストール：

```bash
pip install numpy scikit-learn matplotlib
```

## 使用方法

### テストの実行

```bash
python reservoir_computing.py
```

または、シェルスクリプトで実行：

```bash
bash run_tests.sh
```

### 結果

実行後、以下のファイルが生成されます：
- `reservoir_computing_results.png` - 予測結果の可視化

## テストデータセット

### Test 1: Mackey-Glass時系列
- **説明**: カオス的な非線形時系列
- **サンプル数**: 800 (訓練: 600, テスト: 200)
- **タスク**: 次のステップの値を予測

### Test 2: Sin-Cos合成データ
- **説明**: sin(t) + 0.5 * cos(t)
- **サンプル数**: 400 (訓練: 300, テスト: 100)
- **タスク**: 合成信号の次のステップを予測

## 評価指標

- **MSE** (Mean Squared Error): 二乗平均誤差
- **RMSE** (Root Mean Squared Error): 二乗平均誤差の平方根
- **MAE** (Mean Absolute Error): 平均絶対誤差
- **NMSE** (Normalized MSE): 正規化された二乗平均誤差
- **Correlation**: 相関係数

## 実装詳細

### ReservoirComputerクラス

```python
from reservoir_computing import ReservoirComputer

# モデルの初期化
rc = ReservoirComputer(
    input_size=1,
    reservoir_size=200,
    spectral_radius=0.95,
    sparsity=0.9,
    input_scale=0.5,
    ridge_alpha=1e-6,
    random_state=42
)

# 訓練
rc.train(X_train, y_train, warmup=50)

# 予測
predictions = rc.predict(X_test, warmup=50)
```

### パラメータ

- `input_size`: 入力の次元数
- `reservoir_size`: リザーバーのニューロン数 (デフォルト: 300)
- `spectral_radius`: リザーバーの固有値の最大値 (デフォルト: 0.9)
- `sparsity`: リザーバーの疎性（0で密、1で完全疎） (デフォルト: 0.9)
- `input_scale`: 入力重みのスケーリング (デフォルト: 0.5)
- `ridge_alpha`: Ridge回帰の正則化パラメータ (デフォルト: 1e-6)
- `random_state`: 乱数シード

## パフォーマンス例

### Mackey-Glass時系列
```
MSE        : 0.001234
RMSE       : 0.035123
MAE        : 0.028456
NMSE       : 0.045678
Correlation: 0.987654
```

### Sin-Cos合成データ
```
MSE        : 0.000567
RMSE       : 0.023812
MAE        : 0.018934
NMSE       : 0.021345
Correlation: 0.995123
```

## 参考文献

- Echo State Networks (ESN): Jaeger, H. (2001)
- リザーバーコンピューティングの理論と応用

## ライセンス

MIT License

## 著者

nabemako
