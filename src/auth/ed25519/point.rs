//! Internal Ed25519 point arithmetic.
//!
//! This is the portable extended Edwards baseline using complete addition and
//! a correctness-first scalar-multiplication path.

use core::fmt;

use super::field::FieldElement;

const EDWARDS_D: FieldElement = FieldElement::from_limbs([
  929_955_233_495_203,
  466_365_720_129_213,
  1_662_059_464_998_953,
  2_033_849_074_728_123,
  1_442_794_654_840_575,
]);
const EDWARDS_D2: FieldElement = FieldElement::from_limbs([
  1_859_910_466_990_425,
  932_731_440_258_426,
  1_072_319_116_312_658,
  1_815_898_335_770_999,
  633_789_495_995_903,
]);
const BASEPOINT_X: FieldElement = FieldElement::from_limbs([
  1_738_742_601_995_546,
  1_146_398_526_822_698,
  2_070_867_633_025_821,
  562_264_141_797_630,
  587_772_402_128_613,
]);
const BASEPOINT_Y: FieldElement = FieldElement::from_limbs([
  1_801_439_850_948_184,
  1_351_079_888_211_148,
  450_359_962_737_049,
  900_719_925_474_099,
  1_801_439_850_948_198,
]);

/// Precomputed table: `BASEPOINT_TABLE[i] = i * B` for `i` in `0..16`.
///
/// Enables 4-bit windowed scalar multiplication of the basepoint, reducing
/// point additions from ~128 to ~60 per scalar mul.
#[rustfmt::skip]
const BASEPOINT_TABLE: [ExtendedPoint; 16] = [
  // 0*B
  ExtendedPoint {
    x: FieldElement::from_limbs([0, 0, 0, 0, 0]),
    y: FieldElement::from_limbs([1, 0, 0, 0, 0]),
    z: FieldElement::from_limbs([1, 0, 0, 0, 0]),
    t: FieldElement::from_limbs([0, 0, 0, 0, 0]),
  },
  // 1*B
  ExtendedPoint {
    x: FieldElement::from_limbs([199570966926459, 81994479920299, 1528071091047542, 2249056567190523, 99289794829204]),
    y: FieldElement::from_limbs([450359962737049, 900719925474099, 1801439850948198, 1351079888211148, 450359962737049]),
    z: FieldElement::from_limbs([2251799813685233, 2251799813685247, 2251799813685247, 2251799813685247, 2251799813685247]),
    t: FieldElement::from_limbs([610016736278213, 65595583936239, 772096910100984, 1348885291015369, 529791798600413]),
  },
  // 2*B
  ExtendedPoint {
    x: FieldElement::from_limbs([1808670656391269, 48880706398763, 1977980474703394, 1867776965675273, 967057278898026]),
    y: FieldElement::from_limbs([1347625255176274, 1867032737574310, 1880207660701081, 1287292155099302, 964953763535664]),
    z: FieldElement::from_limbs([1601710909317457, 1798161902658650, 1580466699017983, 1433681117886615, 533169091813902]),
    t: FieldElement::from_limbs([1264683934652617, 158854318361189, 2113194182911296, 1504009319347946, 2091309136161165]),
  },
  // 3*B
  ExtendedPoint {
    x: FieldElement::from_limbs([1917312290058856, 1924054456674346, 2129923340284722, 277596668956314, 2141677320273575]),
    y: FieldElement::from_limbs([139983509564295, 785554773536408, 127995959603144, 616168384098331, 1897260711053713]),
    z: FieldElement::from_limbs([521149832166950, 1031643875719051, 196083306135826, 1629958890639648, 34381351276098]),
    t: FieldElement::from_limbs([1940133901548972, 18467950747690, 1835550909250723, 1934619764050511, 419442050064257]),
  },
  // 4*B
  ExtendedPoint {
    x: FieldElement::from_limbs([1370563369409750, 609540200492159, 782894086765265, 1291813085041758, 972762839457241]),
    y: FieldElement::from_limbs([1700214076623896, 1553802147721107, 1421930702714000, 625874209986379, 1358933388687252]),
    z: FieldElement::from_limbs([523921129317781, 1631994578350710, 1201294073877250, 1453951046758879, 1835147246765522]),
    t: FieldElement::from_limbs([2014471160876480, 1782710510108593, 704920995653856, 2072653853551347, 1778309740598750]),
  },
  // 5*B
  ExtendedPoint {
    x: FieldElement::from_limbs([118572613671191, 36433378901075, 136162628048685, 157549429892213, 1586539629481454]),
    y: FieldElement::from_limbs([1859835490296812, 838054292688781, 1098728403160741, 1031324976696323, 577026965366322]),
    z: FieldElement::from_limbs([1627731852370680, 391022622471604, 2015649996552366, 1767207556700412, 57153473164566]),
    t: FieldElement::from_limbs([945511387950761, 1141689095412143, 1648504005092266, 896181726739522, 473657153883647]),
  },
  // 6*B
  ExtendedPoint {
    x: FieldElement::from_limbs([130524460769239, 992520653759589, 141479531118261, 989348021514007, 160487868128449]),
    y: FieldElement::from_limbs([2022150382921531, 1021007853232837, 765689078503573, 1950186150626816, 299352554669956]),
    z: FieldElement::from_limbs([685344417970498, 1143549197620558, 390342433584456, 952081437525214, 1218137370516959]),
    t: FieldElement::from_limbs([1691598170583389, 1857431558433038, 1883886115061594, 734058415455736, 1350223741607975]),
  },
  // 7*B
  ExtendedPoint {
    x: FieldElement::from_limbs([396019236214677, 1085496200592378, 1883020321158317, 857646031363936, 393729393938109]),
    y: FieldElement::from_limbs([1593874624579879, 1072667753411439, 1571673313273568, 1159503342186806, 693737408396450]),
    z: FieldElement::from_limbs([1456669533275930, 1578851475205458, 2047955278303594, 1830725152257024, 870226664562544]),
    t: FieldElement::from_limbs([856218620311433, 168589956220916, 116267667006712, 1834998821991705, 1689747341954110]),
  },
  // 8*B
  ExtendedPoint {
    x: FieldElement::from_limbs([7367335290916, 40056844091292, 1768655320886709, 762096329428645, 1059851309295364]),
    y: FieldElement::from_limbs([953827410133504, 463622256623175, 1279068110478020, 1902446528586037, 1747893392330145]),
    z: FieldElement::from_limbs([62170688308433, 1672172603187346, 1334932975117362, 1006371796062564, 1164706291359240]),
    t: FieldElement::from_limbs([783151721167720, 1939618870395907, 332984989693667, 895136342199368, 2181525731233071]),
  },
  // 9*B
  ExtendedPoint {
    x: FieldElement::from_limbs([1004011744013765, 388871725361858, 837318189629212, 202918753897451, 1392857372428392]),
    y: FieldElement::from_limbs([1707311135697427, 563292351400593, 2095987300823759, 817512242340779, 1198292849546092]),
    z: FieldElement::from_limbs([149374563674995, 1671979095694890, 2029430390175445, 1382387064544590, 1471471021008861]),
    t: FieldElement::from_limbs([726095502192868, 1271993541058948, 1196485150211216, 444216229460917, 302762201251168]),
  },
  // 10*B
  ExtendedPoint {
    x: FieldElement::from_limbs([2121326959880612, 1412175717566155, 671350087208097, 852177855532815, 22551853310990]),
    y: FieldElement::from_limbs([1011179581718127, 2097458322854927, 2156085325806974, 2122596671479514, 2043973044314064]),
    z: FieldElement::from_limbs([1060692894517199, 940299876281914, 1194725994271137, 2239436033613287, 1311437633147938]),
    t: FieldElement::from_limbs([313397953181974, 2145978460726980, 165958472904060, 29207499197609, 1267512972503862]),
  },
  // 11*B
  ExtendedPoint {
    x: FieldElement::from_limbs([880826961295472, 710503267363125, 1820366053647283, 1001413570185446, 258063921889714]),
    y: FieldElement::from_limbs([72047613003597, 2250199356744230, 1138006337585871, 767930902960587, 338632604133638]),
    z: FieldElement::from_limbs([2232187799401210, 589540627446253, 1009923141145036, 1573552330778783, 1569077073622009]),
    t: FieldElement::from_limbs([1474212689302085, 137897101161133, 769045649324598, 1519984256014844, 1100574269738056]),
  },
  // 12*B
  ExtendedPoint {
    x: FieldElement::from_limbs([656984869322657, 2177179059976715, 818154973771533, 636105091250661, 2083420612468729]),
    y: FieldElement::from_limbs([2031644485366468, 1144778530727395, 445214553210464, 1419828636808322, 1817883012743269]),
    z: FieldElement::from_limbs([651079489088060, 1806176980270883, 1247045966204018, 502667698917425, 611469034967616]),
    t: FieldElement::from_limbs([2011154736360224, 1356770172439335, 1310305829078477, 122256255253652, 382756432701984]),
  },
  // 13*B
  ExtendedPoint {
    x: FieldElement::from_limbs([895271174857617, 670033861840336, 1496566479561340, 18112064503280, 1842288033497791]),
    y: FieldElement::from_limbs([550972140110667, 1742495029227797, 299791415369152, 905709075554181, 65088980327717]),
    z: FieldElement::from_limbs([1595200959141157, 1431051669341538, 1744301759788269, 1414252002641447, 1741929429097241]),
    t: FieldElement::from_limbs([1144082409004080, 1508801828695933, 1191966276621354, 2111413592349897, 547461071376067]),
  },
  // 14*B
  ExtendedPoint {
    x: FieldElement::from_limbs([1461427411921962, 578307291685435, 918573284220639, 1315669618317892, 133380356135561]),
    y: FieldElement::from_limbs([1742416521240520, 2065523018854574, 784635772243321, 1185368998581178, 525054460877640]),
    z: FieldElement::from_limbs([152234468759545, 655567901506303, 87273290563143, 167486121219574, 1507602221752318]),
    t: FieldElement::from_limbs([2161560928102595, 271110815300842, 519004433357610, 621433507014976, 1444721509363611]),
  },
  // 15*B
  ExtendedPoint {
    x: FieldElement::from_limbs([1505223126160407, 241830900301004, 1233734064652176, 249684706158099, 1623207068049509]),
    y: FieldElement::from_limbs([1290008561225070, 2051992141482997, 1687314552776568, 1711887268913591, 600351728488819]),
    z: FieldElement::from_limbs([989133528322061, 1578580954796152, 2241076042760522, 1078766185077077, 330629063735596]),
    t: FieldElement::from_limbs([1719968801781208, 1605275981428558, 1374878595720574, 1475250988206887, 1646066493158885]),
  },
];

/// Internal extended Edwards point `(X, Y, Z, T)`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExtendedPoint {
  x: FieldElement,
  y: FieldElement,
  z: FieldElement,
  t: FieldElement,
}

impl ExtendedPoint {
  /// Extended-coordinate identity point.
  #[must_use]
  pub(crate) const fn identity() -> Self {
    Self {
      x: FieldElement::ZERO,
      y: FieldElement::ONE,
      z: FieldElement::ONE,
      t: FieldElement::ZERO,
    }
  }

  /// Construct an extended point from affine coordinates.
  #[must_use]
  pub(crate) fn from_affine(x: FieldElement, y: FieldElement) -> Self {
    Self {
      x,
      y,
      z: FieldElement::ONE,
      t: x.mul(&y),
    }
  }

  /// Add two extended Edwards points.
  #[must_use]
  pub(crate) fn add(&self, rhs: &Self) -> Self {
    let a = self.y.sub(&self.x).mul(&rhs.y.sub(&rhs.x));
    let b = self.y.add(&self.x).mul(&rhs.y.add(&rhs.x));
    let c = self.t.mul(&rhs.t).mul(&EDWARDS_D2);
    let zz = self.z.mul(&rhs.z);
    let d = zz.add(&zz);
    let e = b.sub(&a);
    let f = d.sub(&c);
    let g = d.add(&c);
    let h = b.add(&a);

    Self {
      x: e.mul(&f),
      y: g.mul(&h),
      z: f.mul(&g),
      t: e.mul(&h),
    }
  }

  /// Double an extended Edwards point.
  ///
  /// Dedicated `dbl-2008-hwcd` formula for `a = -1`: 4 squarings + 4
  /// multiplications, no curve-constant multiply. The general `add(self)`
  /// path costs 4 squarings + 5 multiplications and can't exploit squaring
  /// symmetry in the compiler.
  #[must_use]
  pub(crate) fn double(&self) -> Self {
    let a = self.x.square();
    let b = self.y.square();
    let zz = self.z.square();
    let c = zz.add(&zz); // 2·Z²
    let d = a.neg(); // a·X² = -X² since a = -1
    let e = self.x.add(&self.y).square().sub(&a).sub(&b); // (X+Y)² - X² - Y²
    let g = d.add(&b);
    let f = g.sub(&c);
    let h = d.sub(&b);

    Self {
      x: e.mul(&f),
      y: g.mul(&h),
      z: f.mul(&g),
      t: e.mul(&h),
    }
  }

  /// Compress the point into the standard Ed25519 encoding.
  #[must_use]
  pub(crate) fn to_bytes(self) -> Option<[u8; 32]> {
    let (x, y) = self.to_affine()?;
    let mut bytes = y.to_bytes();
    if x.is_negative() {
      bytes[31] |= 0x80;
    }
    Some(bytes)
  }

  /// Decode a compressed Ed25519 point.
  #[must_use]
  pub(crate) fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
    let sign = (bytes[31] >> 7) != 0;
    let mut y_bytes = *bytes;
    y_bytes[31] &= 0x7F;
    let y = FieldElement::from_bytes(&y_bytes)?;
    let y2 = y.square();
    let numerator = y2.sub(&FieldElement::ONE);
    let denominator = EDWARDS_D.mul(&y2).add(&FieldElement::ONE);
    let x2 = numerator.mul(&denominator.invert()).normalize();
    let mut x = x2.sqrt()?;

    if x.is_zero() && sign {
      return None;
    }
    if x.is_negative() != sign {
      x = x.neg();
    }

    Some(Self::from_affine(x.normalize(), y.normalize()))
  }

  /// Standard Ed25519 basepoint.
  #[must_use]
  pub(crate) fn basepoint() -> Self {
    Self::from_affine(BASEPOINT_X, BASEPOINT_Y)
  }

  /// Scalar multiplication by a little-endian 32-byte scalar.
  #[must_use]
  pub(crate) fn scalar_mul(&self, scalar: &[u8; 32]) -> Self {
    let mut acc = Self::identity();

    for byte in scalar.iter().rev().copied() {
      let mut shift = 8u32;
      while shift > 0 {
        shift = shift.strict_sub(1);
        acc = acc.double();
        if ((byte >> shift) & 1) == 1 {
          acc = acc.add(self);
        }
      }
    }

    acc
  }

  /// Fixed-base multiplication for the Ed25519 basepoint.
  ///
  /// Uses 4-bit windowed scalar multiplication with a precomputed table of
  /// `i*B` for `i` in `0..16`. Processes one nibble per step: 4 doublings
  /// then a table lookup, reducing point additions from ~128 to ~60.
  #[must_use]
  pub(crate) fn scalar_mul_basepoint(scalar: &[u8; 32]) -> Self {
    let mut acc = Self::identity();

    // Process nibbles from most significant to least significant.
    for byte in scalar.iter().rev().copied() {
      let hi = (byte >> 4) as usize;
      let lo = (byte & 0x0F) as usize;

      acc = acc.double().double().double().double();
      if let Some(entry) = BASEPOINT_TABLE.get(hi).filter(|_| hi != 0) {
        acc = acc.add(entry);
      }

      acc = acc.double().double().double().double();
      if let Some(entry) = BASEPOINT_TABLE.get(lo).filter(|_| lo != 0) {
        acc = acc.add(entry);
      }
    }

    acc
  }

  /// Straus/Shamir interleaved double-scalar multiply: `[s]B + [h]A`.
  ///
  /// Combines two scalar multiplications into a single 256-bit scan using
  /// the precomputed basepoint table and a runtime table of 16 multiples of
  /// `a`. Halves the doublings (256 vs 512) compared to two independent
  /// scalar muls.
  ///
  /// Variable-time: branches on scalar nibble values. Safe for verification
  /// where both `s` and `h` are public (derived from the message/signature).
  #[must_use]
  pub(crate) fn straus_basepoint_vartime(s: &[u8; 32], h: &[u8; 32], a: &Self) -> Self {
    // Build runtime table: a_table[i] = i * A.
    let mut a_table = [Self::identity(); 16];
    let mut prev = Self::identity();
    for entry in a_table.iter_mut().skip(1) {
      prev = prev.add(a);
      *entry = prev;
    }

    let mut acc = Self::identity();

    for (&s_byte, &h_byte) in s.iter().zip(h.iter()).rev() {
      // High nibble.
      acc = acc.double().double().double().double();
      let s_hi = (s_byte >> 4) as usize;
      let h_hi = (h_byte >> 4) as usize;
      if let Some(entry) = BASEPOINT_TABLE.get(s_hi).filter(|_| s_hi != 0) {
        acc = acc.add(entry);
      }
      if let Some(entry) = a_table.get(h_hi).filter(|_| h_hi != 0) {
        acc = acc.add(entry);
      }

      // Low nibble.
      acc = acc.double().double().double().double();
      let s_lo = (s_byte & 0x0F) as usize;
      let h_lo = (h_byte & 0x0F) as usize;
      if let Some(entry) = BASEPOINT_TABLE.get(s_lo).filter(|_| s_lo != 0) {
        acc = acc.add(entry);
      }
      if let Some(entry) = a_table.get(h_lo).filter(|_| h_lo != 0) {
        acc = acc.add(entry);
      }
    }

    acc
  }

  /// Multiply by the Edwards cofactor.
  #[must_use]
  pub(crate) fn mul_by_cofactor(&self) -> Self {
    self.double().double().double()
  }

  /// Convert the point to affine coordinates when `Z != 0`.
  #[must_use]
  pub(crate) fn to_affine(self) -> Option<(FieldElement, FieldElement)> {
    if self.z.is_zero() {
      return None;
    }

    let inv_z = self.z.invert();
    Some((self.x.mul(&inv_z).normalize(), self.y.mul(&inv_z).normalize()))
  }

  /// Borrow the extended-coordinate components.
  #[must_use]
  pub(crate) const fn components(&self) -> (&FieldElement, &FieldElement, &FieldElement, &FieldElement) {
    (&self.x, &self.y, &self.z, &self.t)
  }
}

impl Default for ExtendedPoint {
  fn default() -> Self {
    Self::identity()
  }
}

impl fmt::Debug for ExtendedPoint {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ExtendedPoint").finish_non_exhaustive()
  }
}

#[cfg(test)]
mod tests {
  use super::ExtendedPoint;
  use crate::auth::ed25519::{Ed25519SecretKey, field::FieldElement, hash::ExpandedSecret};

  fn basepoint() -> ExtendedPoint {
    ExtendedPoint::basepoint()
  }

  fn decode_hex_32(hex: &str) -> [u8; 32] {
    let bytes = hex.as_bytes();
    let mut out = [0u8; 32];

    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
      *dst = hex_value(chunk[0]) << 4 | hex_value(chunk[1]);
    }

    out
  }

  fn hex_value(byte: u8) -> u8 {
    match byte {
      b'0'..=b'9' => byte - b'0',
      b'a'..=b'f' => byte - b'a' + 10,
      b'A'..=b'F' => byte - b'A' + 10,
      _ => panic!("invalid hex"),
    }
  }

  #[test]
  fn identity_has_expected_affine_coordinates() {
    let affine = ExtendedPoint::identity().to_affine();

    assert_eq!(affine, Some((FieldElement::ZERO, FieldElement::ONE)));
  }

  #[test]
  fn affine_constructor_sets_t_to_xy() {
    let point = basepoint();
    let (x, y, z, t) = point.components();

    assert_eq!(*z, FieldElement::ONE);
    assert_eq!(*t, x.mul(y));
  }

  #[test]
  fn identity_is_neutral_for_addition() {
    let point = basepoint();
    let identity = ExtendedPoint::identity();

    assert_eq!(point.add(&identity).to_affine(), point.to_affine());
    assert_eq!(identity.add(&point).to_affine(), point.to_affine());
  }

  #[test]
  fn doubling_matches_add_self() {
    let point = basepoint();

    assert_eq!(point.double().to_affine(), point.add(&point).to_affine());
  }

  #[test]
  fn basepoint_roundtrips_compressed_encoding() {
    let expected = decode_hex_32("5866666666666666666666666666666666666666666666666666666666666666");
    let encoded = basepoint().to_bytes();

    assert_eq!(encoded, Some(expected));
    assert_eq!(
      encoded
        .and_then(|bytes| ExtendedPoint::from_bytes(&bytes))
        .and_then(|point| point.to_bytes()),
      Some(expected)
    );
  }

  #[test]
  fn compressed_identity_with_sign_bit_set_is_rejected() {
    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    bytes[31] = 0x80;

    assert_eq!(ExtendedPoint::from_bytes(&bytes), None);
  }

  #[test]
  fn scalar_mul_basepoint_zero_is_identity() {
    let point = ExtendedPoint::scalar_mul_basepoint(&[0u8; 32]);

    assert_eq!(point.to_affine(), Some((FieldElement::ZERO, FieldElement::ONE)));
  }

  #[test]
  fn scalar_mul_basepoint_one_is_basepoint() {
    let mut scalar = [0u8; 32];
    scalar[0] = 1;

    assert_eq!(
      ExtendedPoint::scalar_mul_basepoint(&scalar).to_bytes(),
      basepoint().to_bytes()
    );
  }

  #[test]
  fn cofactor_mul_identity_stays_identity() {
    let point = ExtendedPoint::identity().mul_by_cofactor();

    assert_eq!(point.to_affine(), Some((FieldElement::ZERO, FieldElement::ONE)));
  }

  #[test]
  fn rfc8032_public_key_derivation_matches_vector_1() {
    let secret = Ed25519SecretKey::from_bytes(decode_hex_32(
      "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
    ));
    let expanded = ExpandedSecret::from_secret_key(&secret);
    let public = ExtendedPoint::scalar_mul_basepoint(expanded.scalar_bytes()).to_bytes();
    let expected = decode_hex_32("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");

    assert_eq!(public, Some(expected));
  }

  #[test]
  fn basepoint_table_matches_sequential_adds() {
    let b = ExtendedPoint::basepoint();
    let mut acc = ExtendedPoint::identity();
    for (i, entry) in super::BASEPOINT_TABLE.iter().enumerate() {
      assert_eq!(acc.to_affine(), entry.to_affine(), "BASEPOINT_TABLE[{i}] mismatch");
      acc = acc.add(&b);
    }
  }
}
