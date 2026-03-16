# Triangram

三角形(Delaunay分割)を組み合わせて元画像に近似するアート生成プログラム。

## 実行方法

```bash
uv run python main.py
```

出力は `output_images/` ディレクトリに保存される。

## アーキテクチャ

パイプライン構造。各コンポーネントはABCで定義されており、差し替え可能。

```
入力画像
  ↓
Initializer  → 頂点座標リスト(N, 2)を生成
  ↓
Renderer     → Delaunay分割 → 三角形を元画像の平均色で塗り潰し → 画像
  ↓
Evaluator    → MSEでLossを計算(小さいほど良い)
  ↓
Optimizer    × Nフェーズ → 頂点を動かしてLossを下げる(ヒルクライム or 焼きなまし法)
```

### ファイル構成

```
triangram-python/
├── main.py                  # エントリーポイント・実行設定
├── triangram/
│   ├── __init__.py
│   ├── state.py             # TriangramState(頂点・レンダリング結果を保持)
│   ├── base.py              # ABC定義(BaseInitializer/Renderer/Evaluator/Optimizer/Recorder)
│   ├── initializers.py      # RandomInitializer, EdgeAwareInitializer
│   ├── renderers.py         # DelaunayRenderer
│   ├── evaluators.py        # MSEEvaluator, SSIMEvaluator, WeightedEvaluator
│   ├── optimizers.py        # SimpleRandomOptimizer
│   ├── recorders.py         # AnimationRecorder
│   └── pipeline.py          # TriangramPipeline
└── pyproject.toml
```

### 設計方針

- Optimizerのみ複数化（フェーズ順実行のため `add_optimizer()` でリスト化）
- Rendererの複数化は `CompositeRenderer`、Evaluatorの複数化は `WeightedEvaluator` で対応する
  - いずれも「出力が1つに集約される」ため、パイプライン側を変えずに1クラスとして差し替えられる

## TODO

### Bug Fix
- [x] `DelaunayRenderer`: `float → int32` 切り捨てによる隣接三角形間ギャップの解消
  - `triangle_pts.astype(np.int32)` を `np.round(...).astype(np.int32)` に変更
- [x] `DelaunayRenderer`: `LINE_AA` による隣接三角形間の黒い継ぎ目の解消
  - `LINE_AA` はエッジを黒背景とブレンドするため継ぎ目が生じる → `LINE_8` に変更
  - ジャギーはスーパーサンプリング (`supersample` パラメータ) で対処
- [x] `pipeline.target_image` と `state.target_image` が重複している → `self.state.target_image` に統一して前者を削除する
- [x] `setup()` 未コール時のエラーメッセージが不明確 → `run()` の冒頭で明示的にチェックする

### Initializer
- [x] `EdgeAwareInitializer`: Cannyエッジ検出結果をもとにエッジ上に優先的に点を配置

### Optimizer
- [x] `SimulatedAnnealingOptimizer`: 焼きなまし法(確率的に悪化を許容して局所最適を脱出)
- [x] `AdaptiveRefiner`: 誤差駆動の適応的点追加・削除
  - Split: 誤差加重値 (MSE × 三角形面積) が最大の三角形の重心に点を追加
  - Merge: 周辺三角形の誤差合計が最小の点を削除
- [ ] `AdaptiveRefiner` のハイブリッド戦略: 幾何学的基準との組み合わせ
  - 近接点統合: 最近傍点との距離が閾値以下の点を除去
  - 疎領域への追加: 面積の大きい三角形への点追加（誤差によらない補完）
- [ ] 収束判定によるEarly Stopping: 直近Nステップでベストが更新されなければ早期終了 (`iterations` を上限として扱う)

### Evaluator
- [x] `SSIMEvaluator`: 構造的類似度(SSIM)ベースの評価
- [x] `WeightedEvaluator`: 複数Evaluatorの加重合成

### Performance
- [ ] 差分再描画: 変更頂点に隣接する三角形のみ再描画して高速化

### Output
- [x] 最適化過程をGIF/動画として書き出す
