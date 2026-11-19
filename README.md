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

## WebAssembly

`attic-wasm` は、JavaScript から USI コマンドを送って使う WASI モジュールです。
`wasm32-wasip1` 版は呼び出し元で探索を完了します。探索中に `stop` を送れないため、`go depth` や `go nodes` のように終了条件を指定してください。
`wasm32-wasip1-threads` 版は Worker で動き、`go infinite`、`stop`、`Threads` を使えます。ブラウザーでこの版を使うには、クロスオリジン分離を有効にして `SharedArrayBuffer` を利用できるようにしてください。両版とも WebAssembly SIMD に対応した環境が必要です。

以下は評価ファイル不要の `hce_material_17` を使うビルド例です。

```sh
rustup target add wasm32-wasip1 wasm32-wasip1-threads
cargo build --locked --release -p attic-wasm --target wasm32-wasip1 --features hce_material_17
cargo build --locked --release -p attic-wasm --target wasm32-wasip1-threads --features hce_material_17
npm --prefix wasm/core ci
npm --prefix wasm/core run build
```

`wasm/core/dist/index.js` の `createEngine` に、各ターゲットの `release/attic_wasm.wasm` を読み込んだバイト列を渡します。単一スレッド版にはコンパイル済みの `WebAssembly.Module` も渡せます。
NNUE などを使う場合は評価関数に対応する Cargo feature を選び、`files: { "/eval/nn.bin": weights }` を渡します。`weights` は評価ファイルのバイト列です。

```js
import { readFile } from "node:fs/promises";
import { createEngine } from "./wasm/core/dist/index.js";

const wasm = await readFile("./target/wasm32-wasip1/release/attic_wasm.wasm");
const engine = await createEngine({ wasm, print: console.log });
engine.addLineListener((line) => {
  if (line === "readyok") {
    engine.send("position startpos");
    engine.send("go depth 3");
  }
});
try {
  engine.send("isready");
} finally {
  await engine.dispose();
}
```

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

### `sfnn_halfkahm2_threat_1024_15_64_8gp`

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

**探索**は、差分更新を使うビルドと、毎手累積値を作り直すビルド（`threat_force_refresh`）で、結果が一致しました。Sirius の局面集合から取った 584 局面を `go depth 9`（Threads 1、USI_Hash 16）で探索し、`info` 行（`time`・`nps`・`hashfull` を除く）と `bestmove` の計 5,648 行を比べています。同じバイナリを 2 回走らせても一致するので、探索の再現性は崩れていません。`scripts/sirius_search_check.py` で再現できます。

**速度**は、上の 584 局面（depth 9、Threads 1）を、このマシンの P コア 1 つに固定して 3 回測ると、644,520〜693,304 nps（約 674 万ノード）でした。測定中、このマシンでは他のプロセスも動いていました。差分更新を使わず毎手作り直すビルドは、同じ条件で 132,875 と 187,754 nps で、ばらつきが大きく、比としては使えません。YaneuraOu との比は測っていないので、`探索速度` の表には入れていません。

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
| SFNN | `halfkahm2_threat_1024_15_64_8gp`     |
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
| SYNN | `token_128x2_32_32_pair16_o1_r4`      |
| SYNN | `token_128x2_32_32_pair16_o1_r2`      |
| SYNN | `token_128x2_32_32_pair16_o1_hc`      |
| SYNN | `token_128x2_32_32_pair16_o1_r4hc`    |
| SYNN | `token_128x2_32_32_pair16_m8_f2`      |
| HCE  | `material_17`                         |
| HCE  | `mobility_107_57`                     |
