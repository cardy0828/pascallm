//go:build cgo

package main

// cgoEnabled は、このバイナリが cgo を有効にしてビルドされたかを返します。
// この版は CGO_ENABLED=1 のビルドで使われ、常に true です（M1a の目的からは外れる構成）。
func cgoEnabled() bool { return true }
