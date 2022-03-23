## 虚血性心疾患診断システム[1]



## 背景

日本における令和 3 年の全死亡者の死因の 15.0%が心疾患であり，悪性新 生物 (癌) に次いで二番目に高い割合となっている．この心疾患を診断する際には，患者の心臓の形状と動き情報 (動的心臓形状情報) が重要となる．世界的な健康問題である心疾患の多くは虚血性心疾患である．

虚血性心疾患は，生活習慣病の一つとして知られており，糖尿病，喫煙， 高血圧症，高脂血症が 4 大危険因子として知られている．虚血性心疾患の診断や早期発見のためには，心疾患全般の診断に有用な心臓形状情報のみではなく，患者の生活習慣情報も考慮することが重要である．

## 提案手法

心臓モデルと生活習慣情報から虚血性心疾患患者か健常者かを分類する，虚血性心疾患の診断 システムを提案した。提案手法は、心臓モデルからの特徴抽出部と 疾患予測部からなる.特徴抽出部では，心臓モデルから潜在変数 $z_3$ を求める.次に，特徴抽出部で求めた 潜在変数 $z_3$と生活習慣情報を表す生活習慣情報ベクトル $x$を結合し，予測部に入力することで，疾患か健常かの分類結果を出力する.

交差検証の結果として，提案手法は0.8以上の精度で疾患か健常かの分類ができる．特徴的なのは，心疾患の識別の根拠となる画像特徴を潜在空間で確認できることです．

## モデルの構造

![image-20220323163044550](/Users/tianweiye/Library/Application%20Support/typora-user-images/image-20220323163044550.png)

提案手法の構造を図に示す．特徴抽出部は，図 3.4 に示すように，2.1.2 節で示した LVAE[1,2] と，3D 畳み込みエンコーダ・デコーダか らなる. 予測部は，ノード数が 36,36,1である 2 層の多層パーセプトロン (MLP) からなる.

## データの詳細

- 内容：
  心臓の短軸像(Cardiac Short Axis MRI)
  生活習慣データ(心拍数、血圧、喫煙・飲酒頻度など)
- データ数：
  学習データ 150人分(患者:健常者=1:1)
  テストデータ 50人分(患者:健常者=1:1)

## 実験結果

- 混同行列
  ![混同行列](/Users/tianweiye/Downloads/%E5%BC%95%E3%81%8D%E7%B6%99%E3%81%8D%E3%82%99/shuron/results/%E6%B7%B7%E5%90%8C%E8%A1%8C%E5%88%97.png)

- 精度について

  |             | Train | TEST |
  | ----------- | ----- | ---- |
  | Accuracy    | 0.97  | 0.87 |
  | Recall      | 0.89  | 0.86 |
  | Specificity | 0.98  | 0.90 |

  

- 潜在空間と心臓の形状動き特徴

  ![画像特徴](/Users/tianweiye/Downloads/%E5%BC%95%E3%81%8D%E7%B6%99%E3%81%8D%E3%82%99/shuron/results/%E7%94%BB%E5%83%8F%E7%89%B9%E5%BE%B4.png)
  この潜在空間から復元される心臓形状を観察することで，どのような心臓形状を有し ている場合，今後虚血性心疾患を発症しやすいかを予測できる可能性もある．

## 参考文献

[1] 田偉業(2022), "動的心臓形状・生活習慣情報に基づく虚血性心疾患診断システムの構築", 九州大学 システム情報科学研究院 情報知能工学部門修士論文 (未公刊資料)
[2] CK Sønderby, T Raiko, L Maaløe, SK Sønderby, O Winther. Ladder Variational Autoencoders, NIPS 2016
[3] 田 偉業, 宮内 翔子, 諸岡 健一, 倉爪 亮 ,"Ladder Variational Autoencoderを用いた動的心臓形状の特徴量抽出",第40回SICE九州支部学術講演会, 2021