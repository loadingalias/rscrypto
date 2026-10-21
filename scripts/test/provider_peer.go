// Independent Go 1.27.3 crypto/tls peer for provider_interop.py.
package main

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var payload = []byte("rscrypto provider interoperability\n")

func exchange(conn *tls.Conn, client bool, suite uint16, group tls.CurveID) error {
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
		return err
	}
	if err := conn.Handshake(); err != nil {
		return err
	}
	state := conn.ConnectionState()
	if state.Version != tls.VersionTLS13 || state.CipherSuite != suite || state.CurveID != group ||
		state.NegotiatedProtocol != "provider-test" || len(state.VerifiedChains) == 0 {
		return fmt.Errorf("negotiated profile or mutual authentication mismatch: %+v", state)
	}
	for round := 0; round < 2; round++ {
		if client {
			if _, err := conn.Write(payload); err != nil {
				return err
			}
		}
		received := make([]byte, len(payload))
		if _, err := io.ReadFull(conn, received); err != nil {
			return err
		}
		if !bytes.Equal(received, payload) {
			return fmt.Errorf("peer payload mismatch")
		}
		if !client {
			if _, err := conn.Write(payload); err != nil {
				return err
			}
		}
	}
	exported, err := state.ExportKeyingMaterial("provider-interop", nil, 32)
	if err != nil {
		return err
	}
	resumption := "Full"
	if state.DidResume {
		resumption = "Resumed"
	}
	fmt.Printf("RESULT %s %04x %04x %x\n", resumption, state.CipherSuite, uint16(state.CurveID), exported)
	// The provider endpoint sends close_notify after the second exchange.
	var end [1]byte
	if n, err := conn.Read(end[:]); n != 0 || err != io.EOF {
		return fmt.Errorf("expected TLS close_notify: n=%d err=%v", n, err)
	}
	return nil
}

func run() error {
	if len(os.Args) != 8 {
		return fmt.Errorf("usage: peer client|server ADDRESS FIXTURES CREDENTIAL SUITE GROUP COUNT")
	}
	role, address, directory, kind := os.Args[1], os.Args[2], os.Args[3], os.Args[4]
	count, err := strconv.Atoi(os.Args[7])
	if err != nil {
		return err
	}
	endpoint, err := net.ResolveTCPAddr("tcp", address)
	if err != nil {
		return err
	}
	if !endpoint.IP.IsLoopback() || count < 1 || count > 4 {
		return fmt.Errorf("use loopback and 1–4 connections")
	}
	groups := map[string]tls.CurveID{"X25519": tls.X25519, "secp256r1": tls.CurveP256,
		"secp384r1": tls.CurveP384, "X25519MLKEM768": tls.X25519MLKEM768}
	suites := map[string]uint16{"TLS13_AES_128_GCM_SHA256": tls.TLS_AES_128_GCM_SHA256,
		"TLS13_AES_256_GCM_SHA384":       tls.TLS_AES_256_GCM_SHA384,
		"TLS13_CHACHA20_POLY1305_SHA256": tls.TLS_CHACHA20_POLY1305_SHA256}
	group, ok := groups[os.Args[6]]
	if !ok {
		return fmt.Errorf("unknown group")
	}
	suite, ok := suites[os.Args[5]]
	if !ok {
		return fmt.Errorf("unknown suite")
	}
	root, err := os.ReadFile(filepath.Join(directory, kind+"-root.der"))
	if err != nil {
		return err
	}
	leaf, err := os.ReadFile(filepath.Join(directory, kind+"-leaf.der"))
	if err != nil {
		return err
	}
	keyFile := kind + "-leaf-key.der"
	if strings.HasPrefix(kind, "mldsa") {
		// Go's RFC 9881 importer accepts seed-only PKCS#8, not the both form.
		keyFile = kind + "-leaf-seed-key.der"
	}
	key, err := os.ReadFile(filepath.Join(directory, keyFile))
	if err != nil {
		return err
	}
	certificate, err := tls.X509KeyPair(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: leaf}),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key}))
	if err != nil {
		return err
	}
	rootCert, err := x509.ParseCertificate(root)
	if err != nil {
		return err
	}
	roots := x509.NewCertPool()
	roots.AddCert(rootCert)
	config := &tls.Config{MinVersion: tls.VersionTLS13, MaxVersion: tls.VersionTLS13,
		Certificates: []tls.Certificate{certificate}, RootCAs: roots, ClientCAs: roots,
		ClientAuth: tls.RequireAndVerifyClientCert, ServerName: "localhost", NextProtos: []string{"provider-test"},
		CurvePreferences: []tls.CurveID{group}, ClientSessionCache: tls.NewLRUClientSessionCache(4)}
	// Go does not expose TLS 1.3 suite selection. The provider offers one suite;
	// exchange checks that the independently negotiated result matches it.
	switch role {
	case "client":
		for i := 0; i < count; i++ {
			raw, err := net.DialTimeout("tcp", address, 10*time.Second)
			if err != nil {
				return err
			}
			if err := exchange(tls.Client(raw, config), true, suite, group); err != nil {
				return err
			}
		}
	case "server":
		listener, err := net.ListenTCP("tcp", endpoint)
		if err != nil {
			return err
		}
		defer listener.Close()
		fmt.Println("LISTEN", listener.Addr())
		for i := 0; i < count; i++ {
			if err := listener.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
				return err
			}
			raw, err := listener.Accept()
			if err != nil {
				return err
			}
			if err := exchange(tls.Server(raw, config), false, suite, group); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("expected client or server")
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
