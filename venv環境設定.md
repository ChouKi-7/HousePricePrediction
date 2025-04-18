
# Python 仮想環境（venv）の作成・使用手順（macOS + VSCode）

## 1. 仮想環境の作成

プロジェクトルートで以下を実行：

```bash
python3 -m venv venv
```

> `venv/` フォルダが作成され、仮想環境が構築されます。

---

## 2. 仮想環境の有効化（ターミナル）

```bash
source venv/bin/activate
```

> 有効化後、ターミナルに `(venv)` の表示が追加されます。

---

## 3. VSCode 上で仮想環境を選択

1. `Cmd + Shift + P` を押してコマンドパレットを開く  
2. `Python: Select Interpreter` を入力して選択  
3. 表示されたリストから `./venv/bin/python` を選択  
   > ※表示されない場合は、VSCodeを一度再起動してください。

---

## 4. 必要なパッケージのインストール

### ▶︎ `requirements.txt` がある場合：

```bash
pip install -r requirements.txt
```

### ▶︎ 無い場合、自分で手動でインストール：

```bash
pip install pandas numpy matplotlib scikit-learn xgboost joblib
```

> 🔸 その後、以下で `requirements.txt` を出力しておくと便利：

```bash
pip freeze > requirements.txt
```

---

## 5. 仮想環境の終了（必要時）

```bash
deactivate
```
