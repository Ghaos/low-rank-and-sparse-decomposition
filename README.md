# Low Rank and Sparse Decomposition

行列の低ランク-スパース分解を用いて、映像データの前景検出問題を解く。

従来の手法では、データの末端部分（行列における1行目、最終行目、1列目、最終列目に相当）に誤差が生じたため、
提案手法ではそれらを修正する。　　

## ※注意書き
このライブラリは私の修士課程での研究（当初はMATLABで実装）をPythonで書き直したものです。

実験結果や図表は、オリジナルのMATLABで実験したものに基づいています。

以下では、数学的背景を省略して、概要を紹介します。

## 概要

### 前景検出問題
前景検出問題とは、固定カメラで撮影した映像を、背景と前景に分離する問題である。

→参照: [https://en.wikipedia.org/wiki/Foreground_detection](https://en.wikipedia.org/wiki/Foreground_detection)

![](https://lh3.googleusercontent.com/0yEoZLE9NEQwvims20jefglr35GcNJlW9EoJ9P7Npb-hlWS5CouqOqITDNYQ8jH-CIOILGPkneM "LSdecomp")<br/>
*前景検出の例。上図では、道路を横切るトラックが前景として検出されている。*

### Total Variation の導入

動画データに対するTotal Variationとは、

**「時間的・空間的に隣り合うピクセルの変化差分」の和**

で表される。

直観的には、変化が大きいデータほどTotal Variationも大きくなり、全体が均一に近いデータほどTotal Variationは小さくなる。

![](https://lh3.googleusercontent.com/MffyhcMCQGrzYOK2gFa-Qhq5ZbvBO8rKxreI6W-YAYKN_Jp8NV01fZ_AdNWxfbfN8sDGeGkvuco "TotalVariation")<br/>
*左図のようにノイズが多いフレームはTotal Variationは大きくなり、右図では小さくなる。*


### 従来手法の問題点

Total Variationは、各ピクセルと（垂直・水平・時間の3方向に）隣り合うピクセルの差分を計算する。

ここで、従来の手法では、

**（本来隣り合わない）末端のピクセルどうしを隣り合うものとして計算する**

という問題点があった。

![](https://lh3.googleusercontent.com/CR_LtI9Lo4FYwciaCEyMrnlQr_pQqmY8eVa352VLZo6dcfMwyZrQhu9WATP672tHDeBvrdIvEqE "Edges")<br/>
*従来手法では、各フレームの最初と最後の列（図中赤矢印）、最初と最後の行（図中青矢印）、最初と最後のフレーム（図中緑で示したフレーム）をそれぞれ「隣り合う」ものとして計算していた。*

この問題は、行列の計算において循環的な操作が必要だったために生じていた。

そこで、提案手法では循環的な計算を避けることで末端成分の問題点を解消した。

## 実験

### データセット
次の2種類のデータセットに対して実験を行った。
 - 自作データセット
	- 黒背景に白い丸が移動するようなデータ（＋ノイズ）を作成
	- 解像度(160,120)
	 - 180フレーム
 - LIMU Datasets（[http://limu.ait.kyushu-u.ac.jp/dataset/en/index.html](http://limu.ait.kyushu-u.ac.jp/dataset/en/index.html)） - Bus Stop in the Morning
	 - バス停を人や車が移動する映像
	 - 解像度(160,120)にトリミングして使用
	 - 180フレーム（フレーム1655～1834）

### マスクの生成
出力結果は、各ピクセルが連続数で表されるグレースケール画像となる。

これを、ある閾値によって0-1に離散化することで、マスク画像を生成する。

検出結果の評価はこのマスクによってなされる。


### 評価基準

評価基準としてF値を使用した。

→参照: [https://en.wikipedia.org/wiki/F1_score](https://en.wikipedia.org/wiki/F1_score)

F値は、正解データに対して相対的に計算され、完全に一致する場合はその値は1となる。


### 実験結果

合成データに対する結果は下図のようになった。

注目すべきは、従来手法の1フレーム目で、右上に誤検出された領域が確認できる。

これは、1フレーム目が最終フレームに影響を受けたことが原因だと考えられる。

![](https://lh3.googleusercontent.com/_ObOo64weF84uP0ClY91ef3tLfMMsthm619M_RlUUvZJFyFlK6PeFPngAXzhUGBhb7zhMX2_85U "result1")<br/>
*合成データに対する実験結果。1行目が提案手法、2行目が既存手法、3行目が正解データ(Ground Truth)に相当する。F値は、より良かったものを赤字で強調した。*

<br/>

実データに対する結果は下図のようになった。

こちらも、特に1フレーム目でF値の改善が見られた。

![](https://lh3.googleusercontent.com/7D-Lmft8JrwY4iRxfjM5dLDwa7V0W8MUWshsCx8b0H-8xTd9-KvE4us3jFKugbP0uWyE5-i3O9Y "result2")<br/>
*実データに対する実験結果。1行目が提案手法、2行目が既存手法、3行目が正解データ(Ground Truth)に相当する。F値は、より良かったものを赤字で強調した。*


## まとめ

（研究として）

 - Total Vatiationを用いた前景検出において、末端成分における問題を解消した。
 - 合成データと実データで実験を行い、提案手法の優位性を確認した。

（Githubで公開するにあたって）

 - 当初実装したMATLABと同じ結果を返すことを確認した。


## 課題

（研究として）

 - カラー映像に拡張

（Githubで公開するにあたって）

 - 行列の低ランク-スパース分解問題（上記で説明されていない理論的背景）についての説明を加える。
 - MATLABと比べてPythonで大幅に遅くなってしまったので、原因の究明と解決


## 参考文献

 - Xu, Yang, et al. "Low-rank decomposition and total variation regularization of hyperspectral video sequences." _IEEE Transactions on Geoscience and Remote Sensing_ 56.3 (2018): 1680-1694.

