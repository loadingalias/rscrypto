// Independent, C-free quic-go v0.61.0 peer. This is a bounded loopback fixture.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	quic "github.com/quic-go/quic-go"
	"github.com/quic-go/quic-go/qlog"
)

var payload = bytes.Repeat([]byte("rscrypto QUIC application data\n"), 1024)

func credentials(directory, kind string, group tls.CurveID) (*tls.Config, error) {
	root, err := os.ReadFile(filepath.Join(directory, kind+"-root.der"))
	if err != nil {
		return nil, err
	}
	leaf, err := os.ReadFile(filepath.Join(directory, kind+"-leaf.der"))
	if err != nil {
		return nil, err
	}
	keyName := kind + "-leaf-key.der"
	if strings.HasPrefix(kind, "mldsa") {
		keyName = kind + "-leaf-seed-key.der"
	}
	key, err := os.ReadFile(filepath.Join(directory, keyName))
	if err != nil {
		return nil, err
	}
	cert, err := tls.X509KeyPair(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: leaf}), pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key}))
	if err != nil {
		return nil, err
	}
	anchor, err := x509.ParseCertificate(root)
	if err != nil {
		return nil, err
	}
	roots := x509.NewCertPool()
	roots.AddCert(anchor)
	return &tls.Config{MinVersion: tls.VersionTLS13, MaxVersion: tls.VersionTLS13,
		RootCAs: roots, ClientCAs: roots, Certificates: []tls.Certificate{cert},
		ClientAuth: tls.RequireAndVerifyClientCert, ServerName: "localhost", NextProtos: []string{"provider-test"},
		CurvePreferences: []tls.CurveID{group}, ClientSessionCache: tls.NewLRUClientSessionCache(4)}, nil
}

// Use bounded, one-use server-side handles. Default Go stateless tickets can
// put an ML-DSA-87 mutual-authentication chain in the next ClientHello, beyond
// quic-go's fixed 16 KiB CRYPTO stream limit. The provider-server test still
// exercises the provider's stateless ticketer.
func statefulTickets(config *tls.Config) {
	var mu sync.Mutex
	states := make(map[string][]byte)
	config.WrapSession = func(_ tls.ConnectionState, state *tls.SessionState) ([]byte, error) {
		serialized, err := state.Bytes()
		if err != nil {
			return nil, err
		}
		var handle [32]byte
		if _, err := rand.Read(handle[:]); err != nil {
			return nil, err
		}
		mu.Lock()
		defer mu.Unlock()
		if len(states) >= 16 || len(serialized) > 65536 {
			return nil, fmt.Errorf("bounded fixture session cache exhausted")
		}
		states[string(handle[:])] = serialized
		return handle[:], nil
	}
	config.UnwrapSession = func(handle []byte, _ tls.ConnectionState) (*tls.SessionState, error) {
		mu.Lock()
		serialized, ok := states[string(handle)]
		delete(states, string(handle))
		mu.Unlock()
		if !ok {
			return nil, nil
		}
		return tls.ParseSessionState(serialized)
	}
}

func exchange(ctx context.Context, conn *quic.Conn, client bool, resumed, early, reset bool, suite uint16, group tls.CurveID) error {
	for round := 0; round < 2; round++ {
		var stream *quic.Stream
		var err error
		if client {
			stream, err = conn.OpenStreamSync(ctx)
		} else {
			stream, err = conn.AcceptStream(ctx)
		}
		if err != nil {
			return err
		}
		if err := stream.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
			return err
		}
		if client {
			if _, err := stream.Write(payload); err != nil {
				return err
			}
			if err := stream.Close(); err != nil {
				return err
			}
		}
		received, err := io.ReadAll(io.LimitReader(stream, int64(len(payload)+1)))
		if err != nil {
			return err
		}
		if !bytes.Equal(received, payload) {
			return fmt.Errorf("QUIC payload mismatch")
		}
		if !client {
			if _, err := stream.Write(payload); err != nil {
				return err
			}
			if err := stream.Close(); err != nil {
				return err
			}
		}
	}
	select {
	case <-conn.HandshakeComplete():
	case <-ctx.Done():
		return ctx.Err()
	}
	state := conn.ConnectionState()
	tlsState := state.TLS
	if state.Version != quic.Version1 || tlsState.Version != tls.VersionTLS13 || tlsState.CipherSuite != suite || tlsState.CurveID != group || tlsState.NegotiatedProtocol != "provider-test" || len(tlsState.VerifiedChains) == 0 || tlsState.DidResume != resumed || state.Used0RTT != early {
		return fmt.Errorf("negotiation, authentication, resumption, or early-data mismatch: %+v", state)
	}
	exporter, err := tlsState.ExportKeyingMaterial("provider-quic", []byte("interop"), 32)
	if err != nil {
		return err
	}
	kind := "Full"
	if tlsState.DidResume {
		kind = "Resumed"
	}
	fmt.Printf("RESULT %s %04x %04x early=%t %x\n", kind, suite, uint16(group), state.Used0RTT, exporter)
	if reset {
		return nil
	}
	if client {
		return conn.CloseWithError(0, "complete")
	}
	select {
	case <-conn.Context().Done():
	case <-ctx.Done():
		return ctx.Err()
	}
	var application *quic.ApplicationError
	if !errors.As(context.Cause(conn.Context()), &application) || application.ErrorCode != 0 {
		return fmt.Errorf("expected clean peer shutdown: %v", context.Cause(conn.Context()))
	}
	return nil
}

func resetSignal(ctx context.Context, name string, write bool) error {
	directory := os.Getenv("RSCRYPTO_QUIC_RESET_CONTROL")
	if directory == "" {
		return fmt.Errorf("reset scenario needs a control directory")
	}
	path := filepath.Join(directory, name)
	if write {
		return os.WriteFile(path, nil, 0600)
	}
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		if _, err := os.Stat(path); err == nil {
			return nil
		} else if !os.IsNotExist(err) {
			return err
		}
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func run() error {
	if len(os.Args) != 9 {
		return fmt.Errorf("usage: quic-peer client|server ADDRESS FIXTURES CREDENTIAL SUITE GROUP COUNT normal|early|retry|reset")
	}
	address, err := net.ResolveUDPAddr("udp", os.Args[2])
	if err != nil {
		return err
	}
	count, err := strconv.Atoi(os.Args[7])
	if err != nil {
		return err
	}
	if !address.IP.IsLoopback() || count < 1 || count > 4 {
		return fmt.Errorf("use loopback and 1–4 connections")
	}
	groups := map[string]tls.CurveID{"X25519": tls.X25519, "secp256r1": tls.CurveP256, "secp384r1": tls.CurveP384, "X25519MLKEM768": tls.X25519MLKEM768}
	suites := map[string]uint16{"TLS13_AES_128_GCM_SHA256": tls.TLS_AES_128_GCM_SHA256, "TLS13_AES_256_GCM_SHA384": tls.TLS_AES_256_GCM_SHA384, "TLS13_CHACHA20_POLY1305_SHA256": tls.TLS_CHACHA20_POLY1305_SHA256}
	group, ok := groups[os.Args[6]]
	if !ok {
		return fmt.Errorf("unknown group")
	}
	suite, ok := suites[os.Args[5]]
	if !ok {
		return fmt.Errorf("unknown suite")
	}
	mode := os.Args[8]
	if mode != "normal" && mode != "early" && mode != "retry" && !(mode == "reset" && count == 1) {
		return fmt.Errorf("unknown scenario")
	}
	config, err := credentials(os.Args[3], os.Args[4], group)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	qc := &quic.Config{Versions: []quic.Version{quic.Version1}, Allow0RTT: mode == "early", HandshakeIdleTimeout: 5 * time.Second, MaxIdleTimeout: 10 * time.Second, Tracer: qlog.DefaultConnectionTracer}
	switch os.Args[1] {
	case "client":
		for i := 0; i < count; i++ {
			var conn *quic.Conn
			if mode == "early" && i > 0 {
				conn, err = quic.DialAddrEarly(ctx, address.String(), config, qc)
			} else {
				conn, err = quic.DialAddr(ctx, address.String(), config, qc)
			}
			if err != nil {
				return err
			}
			if err := exchange(ctx, conn, true, i > 0, mode == "early" && i > 0, mode == "reset", suite, group); err != nil {
				return err
			}
			if mode == "reset" {
				if err := resetSignal(ctx, "exchanged", true); err != nil {
					return err
				}
				if err := resetSignal(ctx, "ready", false); err != nil {
					return err
				}
				stream, err := conn.OpenUniStreamSync(ctx)
				if err != nil {
					return err
				}
				if _, err := stream.Write([]byte("stateless reset probe")); err != nil {
					return err
				}
				select {
				case <-conn.Context().Done():
				case <-ctx.Done():
					return ctx.Err()
				}
				var reset *quic.StatelessResetError
				if !errors.As(context.Cause(conn.Context()), &reset) {
					return fmt.Errorf("expected authenticated stateless reset: %v", context.Cause(conn.Context()))
				}
				fmt.Println("RESET_RECEIVED")
				if err := resetSignal(ctx, "received", true); err != nil {
					return err
				}
			}
		}
	case "server":
		statefulTickets(config)
		socket, err := net.ListenUDP("udp", address)
		if err != nil {
			return err
		}
		defer socket.Close()
		var reset quic.StatelessResetKey
		if _, err := rand.Read(reset[:]); err != nil {
			return err
		}
		transport := &quic.Transport{Conn: socket, StatelessResetKey: &reset, VerifySourceAddress: func(net.Addr) bool {
			if mode == "retry" {
				fmt.Println("RETRY")
			}
			return mode == "retry"
		}}
		defer transport.Close()
		listener, err := transport.Listen(config, qc)
		if err != nil {
			return err
		}
		defer listener.Close()
		fmt.Println("LISTEN", listener.Addr())
		for i := 0; i < count; i++ {
			conn, err := listener.Accept(ctx)
			if err != nil {
				return err
			}
			if err := exchange(ctx, conn, false, i > 0, mode == "early" && i > 0, mode == "reset", suite, group); err != nil {
				return err
			}
			if mode == "reset" {
				if err := resetSignal(ctx, "exchanged", false); err != nil {
					return err
				}
				// Close the UDP socket first, so shutdown cannot send CONNECTION_CLOSE.
				// A fresh transport retains only the reset key at the same address.
				address := socket.LocalAddr().(*net.UDPAddr)
				if err := socket.Close(); err != nil {
					return err
				}
				if err := transport.Close(); err != nil {
					return err
				}
				replacementSocket, err := net.ListenUDP("udp", address)
				if err != nil {
					return err
				}
				defer replacementSocket.Close()
				replacement := &quic.Transport{Conn: replacementSocket, StatelessResetKey: &reset}
				defer replacement.Close()
				replacementListener, err := replacement.Listen(config, qc)
				if err != nil {
					return err
				}
				defer replacementListener.Close()
				if err := resetSignal(ctx, "ready", true); err != nil {
					return err
				}
				if err := resetSignal(ctx, "received", false); err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("unknown role")
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
