// Exercise the Bouncy Castle 1.87 X.509 provider against both rscrypto composite profiles.
import java.io.ByteArrayInputStream;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.Security;
import java.security.cert.CertPathValidator;
import java.security.cert.CertPathValidatorException;
import java.security.cert.CertificateFactory;
import java.security.cert.PKIXParameters;
import java.security.cert.TrustAnchor;
import java.security.cert.X509Certificate;
import java.time.Instant;
import java.util.Date;
import java.util.List;
import java.util.Set;
import org.bouncycastle.asn1.x500.X500Name;
import org.bouncycastle.asn1.x509.BasicConstraints;
import org.bouncycastle.asn1.x509.ExtendedKeyUsage;
import org.bouncycastle.asn1.x509.Extension;
import org.bouncycastle.asn1.x509.GeneralName;
import org.bouncycastle.asn1.x509.GeneralNames;
import org.bouncycastle.asn1.x509.KeyPurposeId;
import org.bouncycastle.asn1.x509.KeyUsage;
import org.bouncycastle.cert.jcajce.JcaX509v3CertificateBuilder;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.bouncycastle.operator.jcajce.JcaContentSignerBuilder;

class CompositeCertificates {
    private static final Date AT = Date.from(Instant.parse("2027-01-01T00:00:00Z"));
    private static final Date FROM = Date.from(Instant.parse("2026-01-01T00:00:00Z"));
    private static final Date UNTIL = Date.from(Instant.parse("2036-01-01T00:00:00Z"));

    record Profile(String file, String name, String oid) {}
    static final List<Profile> PROFILES = List.of(
        new Profile("mldsa65-ed25519", "MLDSA65-Ed25519-SHA512", "1.3.6.1.5.5.7.6.48"),
        new Profile("mldsa65-ecdsa-p256", "MLDSA65-ECDSA-P256-SHA512", "1.3.6.1.5.5.7.6.45"));

    static X509Certificate read(byte[] der) throws Exception {
        var factory = CertificateFactory.getInstance("X.509", "BC");
        return (X509Certificate) factory.generateCertificate(new ByteArrayInputStream(der));
    }

    static void issue(Profile p, Path dir) throws Exception {
        var generator = KeyPairGenerator.getInstance(p.name(), "BC");
        X500Name issuer = null;
        KeyPair issuerKey = null;
        int serial = 0;
        for (String part : List.of("root", "chain-intermediate", "chain-leaf")) {
            var subject = new X500Name("CN=BC " + p.file() + " " + part);
            var key = generator.generateKeyPair();
            if (issuerKey == null) {
                issuer = subject;
                issuerKey = key;
            }
            var cert = new JcaX509v3CertificateBuilder(issuer, BigInteger.valueOf(++serial),
                FROM, UNTIL, subject, key.getPublic());
            boolean ca = !part.equals("chain-leaf");
            cert.addExtension(Extension.basicConstraints, true, new BasicConstraints(ca));
            cert.addExtension(Extension.keyUsage, true,
                new KeyUsage(ca ? KeyUsage.keyCertSign | KeyUsage.cRLSign : KeyUsage.digitalSignature));
            if (!ca) {
                cert.addExtension(Extension.subjectAlternativeName, false,
                    new GeneralNames(new GeneralName(GeneralName.dNSName, "localhost")));
                cert.addExtension(Extension.extendedKeyUsage, false,
                    new ExtendedKeyUsage(KeyPurposeId.id_kp_serverAuth));
            }
            var signer = new JcaContentSignerBuilder(p.name()).setProvider("BC").build(issuerKey.getPrivate());
            Files.write(dir.resolve(p.file() + "-" + part + ".der"), cert.build(signer).getEncoded(),
                java.nio.file.StandardOpenOption.CREATE_NEW);
            issuer = subject;
            issuerKey = key;
        }
    }

    static void validate(X509Certificate root, X509Certificate intermediate, X509Certificate leaf) throws Exception {
        var chain = CertificateFactory.getInstance("X.509", "BC").generateCertPath(List.of(leaf, intermediate));
        var policy = new PKIXParameters(Set.of(new TrustAnchor(root, null)));
        policy.setDate(AT);
        policy.setRevocationEnabled(false);
        CertPathValidator.getInstance("PKIX", "BC").validate(chain, policy);
    }

    static void verify(Profile p, Path dir) throws Exception {
        var root = read(Files.readAllBytes(dir.resolve(p.file() + "-root.der")));
        var intermediate = read(Files.readAllBytes(dir.resolve(p.file() + "-chain-intermediate.der")));
        var bytes = Files.readAllBytes(dir.resolve(p.file() + "-chain-leaf.der"));
        var leaf = read(bytes);
        for (var cert : List.of(root, intermediate, leaf)) {
            if (!p.oid().equals(cert.getSigAlgOID())) throw new AssertionError("unexpected certificate algorithm");
        }
        root.verify(root.getPublicKey(), "BC");
        validate(root, intermediate, leaf);
        // The final BIT STRING holds the ML-DSA signature followed by the traditional signature.
        for (int index : new int[] {bytes.length - leaf.getSignature().length, bytes.length - 1}) {
            var altered = bytes.clone();
            altered[index] ^= 1;
            var bad = read(altered);
            try {
                validate(root, intermediate, bad);
                throw new AssertionError("accepted an altered leaf signature");
            } catch (CertPathValidatorException expected) {
                if (expected.getIndex() != 0) throw expected;
            }
        }
        System.out.println("BC accepted " + p.name() + " chain and rejected both altered leaf signatures");
    }

    public static void main(String[] args) throws Exception {
        Security.addProvider(new BouncyCastleProvider());
        var dir = Path.of(args[1]);
        for (var profile : PROFILES) {
            if (args[0].equals("issue")) issue(profile, dir);
            verify(profile, dir);
        }
    }
}
