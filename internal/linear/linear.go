// Package linear は、CPU 上の行列 × ベクトル（y = W·x）の共通の形と、FP32 の実装をまとめます。
//
// 参照実装は、重みの形式（FP32・PQ4 など）に関係なく Matrix を通して計算します。
// 行列は「出力次元 × 入力次元」の行優先（PyTorch の nn.Linear の weight と同じ並び）です。
package linear

import (
	"fmt"
	"runtime"
	"sync"
)

// Matrix は、y = W·x を計算できる行列です。
type Matrix interface {
	// Rows は出力次元を返します。例: q_proj なら 2048
	Rows() int
	// Cols は入力次元を返します。例: q_proj なら 2048
	Cols() int
	// MulVec は dst = W·x を計算します。len(dst) == Rows()、len(x) == Cols() であること。
	MulVec(dst, x []float32)
}

// Dense は FP32 の密な行列です。Data は Rows × Cols の行優先です。
type Dense struct {
	rows, cols int
	data       []float32
}

// NewDense は、行優先の data を rows × cols の行列として包みます（コピーしない）。
// 例: NewDense(2048, 2048, weights)
func NewDense(rows, cols int, data []float32) (*Dense, error) {
	if rows <= 0 || cols <= 0 || len(data) != rows*cols {
		return nil, fmt.Errorf("linear: dense %dx%d needs %d values, got %d", rows, cols, rows*cols, len(data))
	}
	return &Dense{rows: rows, cols: cols, data: data}, nil
}

// Rows は出力次元を返します。
func (d *Dense) Rows() int { return d.rows }

// Cols は入力次元を返します。
func (d *Dense) Cols() int { return d.cols }

// Row は index 行目（0 始まり）の重みを返します（コピーしない）。埋め込み表の参照に使います。
func (d *Dense) Row(index int) []float32 {
	return d.data[index*d.cols : (index+1)*d.cols]
}

// MulVec は dst = W·x を、行を CPU コア数で分けて並列に計算します。
func (d *Dense) MulVec(dst, x []float32) {
	CheckShapes(d, dst, x)
	ParallelRows(d.rows, func(begin, end int) {
		for row := begin; row < end; row++ {
			dst[row] = Dot(d.data[row*d.cols:(row+1)*d.cols], x)
		}
	})
}

// Dot は 2 つのベクトルの内積を返します。4 個ずつの部分和にして、加算の依存を減らしています。
// 例: Dot([1, 2, 3], [4, 5, 6]) = 32
func Dot(a, b []float32) float32 {
	var sum0, sum1, sum2, sum3 float32
	length := len(a)
	index := 0
	for ; index+4 <= length; index += 4 {
		sum0 += a[index] * b[index]
		sum1 += a[index+1] * b[index+1]
		sum2 += a[index+2] * b[index+2]
		sum3 += a[index+3] * b[index+3]
	}
	for ; index < length; index++ {
		sum0 += a[index] * b[index]
	}
	return (sum0 + sum1) + (sum2 + sum3)
}

// CheckShapes は、MulVec の引数の長さが行列の形と合わない場合に panic します（呼び出し側の誤り）。
func CheckShapes(m Matrix, dst, x []float32) {
	if len(dst) != m.Rows() || len(x) != m.Cols() {
		panic(fmt.Sprintf("linear: MulVec with dst %d and x %d on a %dx%d matrix", len(dst), len(x), m.Rows(), m.Cols()))
	}
}

// minRowsPerWorker は、1 つのゴルーチンに割り当てる最小の行数です。小さい行列で並列化しすぎないための下限です。
const minRowsPerWorker = 16

// ParallelRows は、0〜rows-1 の行を連続した区間に分け、work(begin, end) を並列に実行して終了を待ちます。
// 例: rows = 2048、12 コアなら、約 171 行ずつ 12 個のゴルーチンに分ける。
func ParallelRows(rows int, work func(begin, end int)) {
	workers := runtime.GOMAXPROCS(0)
	if maxWorkers := (rows + minRowsPerWorker - 1) / minRowsPerWorker; workers > maxWorkers {
		workers = maxWorkers
	}
	if workers <= 1 {
		work(0, rows)
		return
	}
	chunk := (rows + workers - 1) / workers
	var group sync.WaitGroup
	for begin := 0; begin < rows; begin += chunk {
		end := min(begin+chunk, rows)
		group.Add(1)
		go func() {
			defer group.Done()
			work(begin, end)
		}()
	}
	group.Wait()
}
