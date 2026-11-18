# YaneuraOu との互換性

Attic の基準は [YaneuraOu V9.96](https://github.com/yaneurao/YaneuraOu/commit/b707119bbeeeb1972806c3371545d6056b30e061)（`b707119b`）です。

同じ局面・同じ条件で探索したときに、各反復の深さ・評価値・ノード数・読み筋と `bestmove` が YaneuraOu と行単位で一致することを目指しています。確認できた範囲は次のとおりで、これ以外は確認していません。

一致を確かめるのは、YaneuraOu の `make normal`(千日手を 4 回目の出現で判定する構成)に合わせた次のビルドだけです。`ref_check_time` は `go nodes N` の止まる位置を YaneuraOu に合わせます。`quick-draw` は YaneuraOu の `ENABLE_QUICK_DRAW`(`FOR_TOURNAMENT` のビルドで有効)に当たるので、外します。

```sh
cargo build --release -p attic --no-default-features --features <shape>,ref_check_time
```

測り方は、1 局面あたり 10 万ノード、置換表 16 MB、1 スレッドで探索した 16,384 局面のうち、全行(`time` / `nps` / `hashfull` を除く)と `bestmove` が YaneuraOu と一致した局面を数えています。探索ごとに `usinewgame` → `isready` → `position` → `go nodes 100000` を送ります。

## V9.96(`b707119b`)で測ったもの

比較先は YaneuraOu V9.96 の `b707119b`(`make normal TARGET_CPU=AVX2 COMPILER=g++`、`EXTRA_LDFLAGS=-static`)です。Attic は、`ishidakei/main`(`e16990e`)を `develop`(`6e079d9`)にマージした時点のソースで、musl 静的リンク、`-C target-cpu=x86-64-v3` のビルドです。AVX-512 のカーネルは通りません。

| 評価関数                  |   一致 |  総数 |   一致率 |
| ------------------------- | -----: | ----: | -------: |
| `nnue_kp_256x2_32_32`     |  16384 | 16384 | 100.000% |
| `nnue_halfkp_256x2_32_32` |  16384 | 16384 | 100.000% |
| 上記以外                  | 未確認 |     — |        — |

この表は V9.96 追従直後の検証結果です。次の節には V9.80 時点の検証結果を掲載しています。現行の 9 評価関数を使った 1 局面の比較は[パフォーマンス](performance.md#探索結果の照合)を参照してください。

## V9.80(`d47c9bfb`)のときに測ったもの

次の表は **V9.80 に基準を置いていた時点の Attic** で測ったもので、V9.96 では測り直していません(`nnue_kp_256x2_32_32` と `nnue_halfkp_256x2_32_32` の行は、上の V9.96 の表が新しい値です)。比較先は、`nnue_` の 4 行が YaneuraOu V9.80、`sfnn_` の 3 行が shielune 版 YaneuraOu（`8b55d5d`）です。V9.80 には層スタックの選び方がないためで、shielune 版は `least_significant_square_bb` の 1 行を V9.80 の形に戻して使っています。`sfnn_halfkahm2_1024_15_64_progress8ek` の行は、tatara で `progress8ek` の規則で学習したネット（8 superbatch の試作）で比べています。Attic は、`sfnn_` の 3 行が `f5dfd5c`、`nnue_` の 4 行が `770a060` です。

| 評価関数                                   |   一致 |  総数 |   一致率 |
| ------------------------------------------ | -----: | ----: | -------: |
| `nnue_kp_256x2_32_32`                      |  16384 | 16384 | 100.000% |
| `nnue_halfkp_256x2_32_32`                  |  16384 | 16384 | 100.000% |
| `nnue_halfkp_128x2_32_32`                  |  16384 | 16384 | 100.000% |
| `nnue_halfkp_768x2_16_64`                  |  16384 | 16384 | 100.000% |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` |  16384 | 16384 | 100.000% |
| `sfnn_halfkahm2_2304_15_64_progress8ek`    |  16384 | 16384 | 100.000% |
| `sfnn_halfkahm2_1024_15_64_progress8ek`    |  16384 | 16384 | 100.000% |
| 上記以外                                   | 未確認 |     — |        — |

shielune 版の `least_significant_square_bb` を戻さずに比べると、`sfnn_halfkahm2_1024_15_64_progress8kpabs` は 31 局面、`sfnn_halfkahm2_2304_15_64_progress8ek` は 54 局面で食い違います。桂の SEE で桂が 2 枚同時に消える、参照側の挙動によるものです。

## `sfnn_halfkahm2_1024_15_64_progress8ek`

この構成は `sfnn_halfkahm2_1024_15_64_progress8kpabs`（NAGISA V3）と、ファイルの形も進行度の係数も同じで、層スタックの選び方だけが違います。相入玉の局面（先手玉が 1〜5 段、後手玉が 5〜9 段）を 9 番目の層スタックに割り当てます。規則は `sfnn_halfkahm2_2304_15_64_progress8ek` と同じです。どちらの規則で学習したネットかは、ファイルから分かりません。手元の、相入玉の bucket つきで学習したネット（tatara で学習して変換したもの）は、アーキテクチャ名もヘッダと FT のハッシュも V3 と同じでした。使うネットに合う構成を、ビルドで選んでください。

- 配布ディレクトリの `eval_options.txt` に `LS_BUCKET_MODE` の行があっても、attic にはこのオプションがないため、`Error : option name not found : LS_BUCKET_MODE` を出して無視します。規則はビルドで決まります。手元の V3 の配布物は `progress8kpabs`、試作のネットのディレクトリは `progress8ek` と書いていました。

## `sfnn_halfkahm2_threat_1024_15_64_8gp`

Sirius の `nn.bin`（300,147,971 バイト。sha256 は `2d2f8e94…` で始まります）を、そのまま読む構成です。HalfKA_hm2 の特徴量に、盤上の駒どうしの利き（threat 特徴量）を足します。成り駒を含む玉以外の駒がすべて攻め手になり、飛び駒は最初に当たる駒で止まります。層スタックは 8 つで、手駒と敵陣の駒から整数で出す進行度と、玉の位置と守りの駒の数から選びます。境界の近くでは、隣の層スタックとの内分をとります。`FV_SCALE` の既定値は 26 で、評価値は ±31753 に収めます。`progress.bin` は使いません。

ヘッダのハッシュが `0x684f7658` でない、アーキテクチャ文字列の長さが 0 でない、threat のプロファイルが 23 でない、のどれかなら、理由を添えて読み込みを拒否します。他の SFNN の構成と違い、警告を出して先へ進むことはしません。

静的評価の比較先は、Sirius 本体の `e` コマンド（手番側から見た値）です。探索は、Sirius 本体ではなく、同じ評価関数を載せた YaneuraOu と照合しています（下記）。Sirius の探索との一致は確認していません。確認できたのは次のとおりです。

**静的評価**は、4 つの局面集合のすべてで Sirius と一致しました。次の 3 つのビルドのどれでも同じ結果です。

- スカラー: `-C target-cpu=x86-64-v2`（AVX2 なし）
- AVX2: `-C target-cpu=x86-64-v3`
- ネイティブ: このマシン（Core Ultra 7 265KF）の AVX2 と AVX-VNNI。AVX-512 はありません

| 局面集合 | 内容                                  |   一致 |  総数 |
| -------- | ------------------------------------- | -----: | ----: |
| fixture  | 通常の局面                            |  16384 | 16384 |
| probe    | 入玉、駒の改変、駒の山                |   3000 |  3000 |
| packed   | 利きの一覧が 128 件の上限に達する局面 |    500 |   500 |
| moves    | 局面に指し手を進めた後                |   2000 |  2000 |

局面は、8 つの層スタックのすべてを通ります。層スタックごとの局面数は、順に 15,717、2,715、1,002、600、177、27、228、1,418 です。

**差分更新**は、手ごとに、全計算した累積値と一致します。利きの一覧も、全計算で作ったものと行の並びまで一致します。

- Sirius の局面集合の局面と、手で作った局面から始めた、ランダムな対局 477 局、計 40,049 手。取る手 12,021、打つ手 9,064、成る手 3,760、玉の手 11,551（うち、玉が左右反転の境界をまたぐ手が 2,026）を含みます。利きの一覧が上限に達した手が 77、2 つの視点で反転の有無が違う手が 19,940 です。3 つのビルドで同じ件数でした。
- 利きの一覧の（攻め手、対象）の組と並びは、別に書いた素朴な列挙（盤面の配列を走査）とも照合しています。行の番号は、上の Sirius との一致で確かめています。
- moves の 2,000 局面は、指し手を差分更新で進めた先の値も Sirius と一致しました（計 7,022 手）。

**YaneuraOu との探索の一致**は、上流 `b707119b` に Sirius の評価関数（読み込み、threat 特徴量、層スタックの選択と内分、FV_SCALE 26）を足したビルド（`make normal`、`TARGET_CPU=AVX512VNNI`、静的リンク）と、Attic の `--no-default-features --features sfnn_halfkahm2_threat_1024_15_64_8gp,ref_check_time` を、上の表と同じ測り方（16,384 局面、`go nodes 100000`、Threads 1、USI_Hash 16）で突き合わせました。16,384 局面すべてで、全行が一致しました。どちらも Xeon Silver 4210（AVX-512 あり）で実行しています。この YaneuraOu のビルドは上流にはないため、上の表には入れていません。

**探索**は、差分更新を使うビルドと、毎手累積値を作り直すビルド（`threat_force_refresh`）で、結果が一致しました。Sirius の局面集合から取った 584 局面を `go depth 9`（Threads 1、USI_Hash 16）で探索し、`info` 行（`time`・`nps`・`hashfull` を除く）と `bestmove` の計 5,648 行を比べています。同じバイナリを 2 回走らせても一致するので、探索の再現性は崩れていません。[scripts/sirius_search_check.py](../scripts/sirius_search_check.py) で再現できます。

AVX-512 を持つ CPU（Xeon Silver 4210、`-C target-cpu=native`）でも、静的評価の 4 つの局面集合、差分更新の moves 2,000 局面、差分更新と作り直しの探索の比較（398 局面、`go depth 9`）が、すべて一致しました。この構成に AVX-512 のカーネルはなく、AVX-512 のマシンでも AVX2 のカーネルが動きます。

確認していないこと:

- 2 スレッド以上での探索。
- `FV_SCALE` が 26 以外のときの Sirius の値。Sirius と照合したのは 26 の値だけです。26 以外の割り算と ±31753 の丸めは、Attic 自身の出力（`FV_SCALE 1`）と比べています。fixture の最大値は 6,992 で、丸めに届く局面はありません。

Sirius の `nn.bin` と局面ファイルは配布物に含まれません。環境変数 `ATTIC_SIRIUS_REF` がないと、次のテストは通知を出して成功扱いで飛ばします。そのディレクトリには `eval/nn.bin`、`fixture/`、`eval-re/` を置きます。カーネルはコンパイル時に決まるので、スカラーや AVX2 を試すには `RUSTFLAGS` で `target-cpu` を指定します。

```sh
ATTIC_SIRIUS_REF=<sirius-ref> RUSTFLAGS="-C target-cpu=x86-64-v2" \
  cargo test --release -p attic-eval --features sfnn_halfkahm2_threat_1024_15_64_8gp \
  --test sirius_parity --test sirius_incremental --test sirius_fv_scale -- --nocapture
```

## つようら王11 Micro / Mini

配布の評価ファイルを使い、`FV_SCALE=24` でやねうら王の `e` コマンドと照合した結果、Micro / Mini とも静的評価が 2,049 局面すべてで一致しました。1 局面の探索出力の比較は[パフォーマンス](performance.md#探索結果の照合)を参照してください。
