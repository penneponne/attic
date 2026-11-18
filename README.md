# attic-shogi

Rust で書かれた、[YaneuraOu](https://github.com/yaneurao/YaneuraOu) の探索部の互換エンジンです。

基準は [V9.96](https://github.com/yaneurao/YaneuraOu/commit/b707119bbeeeb1972806c3371545d6056b30e061)（`b707119b`）です。以下で YaneuraOu と書いたものは、特に断らない限りこの版を指します。

バージョン番号は、YaneuraOu のどこまでを取り込んだかを表します。このワークスペースは 9.96.0 で、`usi` の `id name` は `Attic 9.96git` です。以下の表のうち、V9.80 で測ったものにはそう書いてあります。

## コマンド

USI プロトコルを標準入出力で話します。引数なしで `attic` を起動すると、コマンドの入力待ちになります。

| コマンド                                                                       | 説明                                                                                                                          |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `usi`                                                                          | エンジン情報とオプション一覧を出力し、`usiok` を返す                                                                          |
| `isready`                                                                      | 定跡と評価関数の `<EvalDir>/nn.bin` を読み込み、`Threads` / `USI_Hash` / `NumaPolicy` を適用して、成功すれば `readyok` を返す |
| `setoption name <名前> value <値>`                                             | オプションを設定する。一覧は `usi` が出力する                                                                                 |
| `usinewgame`                                                                   | 新規対局の開始                                                                                                                |
| `position [startpos \| sfen <SFEN>] [moves <手>]`                              | 局面を設定する                                                                                                                |
| `go`                                                                           | 探索を開始し、`bestmove` を返す                                                                                               |
| `stop`                                                                         | 探索を停止する                                                                                                                |
| `ponderhit`                                                                    | 先読みが的中したことを通知する                                                                                                |
| `gameover`                                                                     | 対局の終了                                                                                                                    |
| `quit`                                                                         | 終了する                                                                                                                      |
| `bench [ttSizeMB] [threads] [limit] [default\|current\|<fenFile>] [limitType]` | 固定条件で NPS を計測する。引数はすべて省略でき、既定値は置換表 1024 MB、1 スレッド、`limit` 15000、`limitType` は `movetime` |

## YaneuraOu との互換性

同じ局面・同じ条件で探索したときに、各反復の深さ・評価値・ノード数・読み筋と `bestmove` が YaneuraOu と行単位で一致することを目指しています。確認できた範囲は次のとおりで、これ以外は確認していません。

一致を確かめるのは、YaneuraOu の `make normal`(千日手を 4 回目の出現で判定する構成)に合わせた次のビルドだけです。`ref_check_time` は `go nodes N` の止まる位置を YaneuraOu に合わせます。`quick-draw` は YaneuraOu の `ENABLE_QUICK_DRAW`(`FOR_TOURNAMENT` のビルドで有効)に当たるので、外します。

```sh
cargo build --release -p attic --no-default-features --features <shape>,ref_check_time
```

測り方は、1 局面あたり 10 万ノード、置換表 16 MB、1 スレッドで探索した 16,384 局面のうち、全行(`time` / `nps` / `hashfull` を除く)と `bestmove` が YaneuraOu と一致した局面を数えています。探索ごとに `usinewgame` → `isready` → `position` → `go nodes 100000` を送ります。

### V9.96(`b707119b`)で測ったもの

比較先は YaneuraOu V9.96 の `b707119b`(`make normal TARGET_CPU=AVX2 COMPILER=g++`、`EXTRA_LDFLAGS=-static`)です。Attic は、`ishidakei/main`(`e16990e`)を `develop`(`6e079d9`)にマージした時点のソース(`crates/*/src` はそれ以降変わっていません)で、musl 静的リンク、`-C target-cpu=x86-64-v3` のビルドです。AVX-512 のカーネルは通りません。

| 評価関数                  |   一致 |  総数 |   一致率 |
| ------------------------- | -----: | ----: | -------: |
| `nnue_kp_256x2_32_32`     |  16384 | 16384 | 100.000% |
| `nnue_halfkp_256x2_32_32` |  16384 | 16384 | 100.000% |
| 上記以外                  | 未確認 |     — |        — |

V9.96 追従のあとに測り直したのはこの 2 行だけです。次の節の表は V9.96 に追従する前の Attic で測ったもので、残りの評価関数が V9.96 でも一致するかは確かめていません。

### V9.80(`d47c9bfb`)のときに測ったもの

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

### `sfnn_halfkahm2_1024_15_64_progress8ek`

この構成は `sfnn_halfkahm2_1024_15_64_progress8kpabs`（NAGISA V3）と、ファイルの形も進行度の係数も同じで、層スタックの選び方だけが違います。相入玉の局面（先手玉が 1〜5 段、後手玉が 5〜9 段）を 9 番目の層スタックに割り当てます。規則は `sfnn_halfkahm2_2304_15_64_progress8ek` と同じです。どちらの規則で学習したネットかは、ファイルから分かりません。手元の、相入玉の bucket つきで学習したネット（tatara で学習して変換したもの）は、アーキテクチャ名もヘッダと FT のハッシュも V3 と同じでした。使うネットに合う構成を、ビルドで選んでください。

- 配布ディレクトリの `eval_options.txt` に `LS_BUCKET_MODE` の行があっても、attic にはこのオプションがないため、`Error : option name not found : LS_BUCKET_MODE` を出して無視します。規則はビルドで決まります。手元の V3 の配布物は `progress8kpabs`、試作のネットのディレクトリは `progress8ek` と書いていました。

## 探索速度

**この節の値は V9.80 に基準を置いていた時点の Attic と YaneuraOu V9.80 で測ったもの**で、V9.96 では測り直していません。V9.96 の追従で探索・置換表・履歴表が変わっているので、速度比は変わっている可能性があります。

YaneuraOu を 1 とした比です。YaneuraOu 側も同じ評価関数で動かしています。表は、ビルドの方法ごとに測ったうち速い方の値です。1 スレッドで、1 局面 10 万ノード × 128 局面を交互に測定しました。

| ビルド                                     | コマンド                | 速度比 |
| ------------------------------------------ | ----------------------- | ------ |
| `nnue_kp_256x2_32_32`                      | `cargo xtask build-pgo` | 1.053  |
| `nnue_halfkp_256x2_32_32`                  | `cargo xtask build-pgo` | 1.022  |
| `nnue_halfkp_128x2_32_32`                  | `cargo xtask build-pgo` | 0.998  |
| `nnue_halfkp_768x2_16_64`                  | `cargo xtask build-pgo` | 1.008  |
| `sfnn_halfkahm2_1024_15_64_progress8kpabs` | `cargo xtask build-pgo` | 0.424  |
| `sfnn_halfkahm2_2304_15_64_progress8ek`    | `cargo xtask build-pgo` | 0.416  |

SFNN の 2 つは、YaneuraOu 側が shielune 版です。遅い原因は調べていません。AVX-512 のカーネルと、表にない評価関数の速度は測っていません。

## 対応している評価関数

| type | feature                               |
| ---- | ------------------------------------- |
| SFNN | `halfkahm2_1536_15_32_k3k3`           |
| SFNN | `halfkahm2_1024_15_64_progress8kpabs` |
| SFNN | `halfkahm2_1024_15_64_progress8ek`    |
| SFNN | `halfkahm2_2304_15_64_progress8ek`    |
| NNUE | `kp_256x2_32_32`                      |
| NNUE | `kp_128x2_32_32`                      |
| NNUE | `halfkp_256x2_32_32`                  |
| NNUE | `halfkp_128x2_32_32`                  |
| NNUE | `halfkp_768x2_16_64`                  |
| SYNN | `token_128x2_32_32`                   |
| SYNN | `token_256x2_32_32`                   |
| SYNN | `kp_128x2_32_32`                      |
| SYNN | `kp_256x2_32_32`                      |
| SYNN | `halfkp_128x2_32_32`                  |
| SYNN | `token_128x2_32_32_pair16`            |
| SYNN | `token_128x2_32_32_pair16_o1`         |
| SYNN | `token_128x2_32_32_pair16_o1_hm`      |
| SYNN | `token_128x2_32_32_pair16_m8_f2`      |
| HCE  | `material_17`                         |
| HCE  | `mobility_107_57`                     |
