//go:build !cgo

package main

// cgoEnabled は、このバイナリが cgo を有効にしてビルドされたかを返します。
// この版は CGO_ENABLED=0 のビルドで使われ、常に false です。
func cgoEnabled() bool { return false }
