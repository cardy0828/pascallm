package server

import (
	"context"
	"errors"
	"github.com/cardy0828/pascallm/internal/generate"
	"testing"
	"time"
)

// TestQueueFIFO は満杯時拒否と受付順の実行を、GPUなしで検証します。
func TestQueueFIFO(t *testing.T) {
	s := New(Config{})
	order := []int{}
	var last chan struct{}
	for i := 0; i < maxQueuedJobs; i++ {
		n := i
		last = make(chan struct{})
		s.jobs <- &job{ctx: context.Background(), done: last, run: func(*generate.Engine) { order = append(order, n) }}
	}
	if err := s.submit(context.Background(), func(*generate.Engine) {}); !errors.Is(err, errQueueFull) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	finished := make(chan struct{})
	go func() { s.RunWorker(ctx, nil); close(finished) }()
	select {
	case <-last:
	case <-time.After(time.Second):
		t.Fatal("worker blocked")
	}
	cancel()
	<-finished
	if len(order) != maxQueuedJobs {
		t.Fatal(order)
	}
	for i, n := range order {
		if i != n {
			t.Fatal(order)
		}
	}
}

// TestQueueCancelled はworker未起動でも待機要求のキャンセルが戻り、実行されないことを確認します。
func TestQueueCancelled(t *testing.T) {
	s := New(Config{})
	ctx, cancel := context.WithCancel(context.Background())
	result := make(chan error, 1)
	go func() { result <- s.submit(ctx, func(*generate.Engine) { t.Error("cancelled work ran") }) }()
	deadline := time.After(time.Second)
	for len(s.jobs) == 0 {
		select {
		case <-deadline:
			t.Fatal("not queued")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cancel()
	select {
	case err := <-result:
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancel blocked")
	}
	queued := <-s.jobs
	if queued.state.Load() != jobCancelled {
		t.Fatal("not cancelled")
	}
	s.jobs <- queued
	workerContext, stopWorker := context.WithCancel(context.Background())
	s.jobs <- &job{ctx: context.Background(), done: make(chan struct{}), run: func(*generate.Engine) { stopWorker() }}
	s.RunWorker(workerContext, nil)

}

// TestQueueWorkerShutdown は停止したworkerへ送った要求が待ち続けないことを確認します。
func TestQueueWorkerShutdown(t *testing.T) {
	s := New(Config{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s.RunWorker(ctx, nil)
	result := make(chan error, 1)
	go func() { result <- s.submit(context.Background(), func(*generate.Engine) { t.Error("worker stopped") }) }()
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("missing stop error")
		}
	case <-time.After(time.Second):
		t.Fatal("shutdown blocked")
	}
}
