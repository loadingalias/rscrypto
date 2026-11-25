use rscrypto::{
  Kem as _, MlKem1024 as Profile, MlKem1024Ciphertext as Ciphertext, MlKem1024DecapsulationKey as DecapsulationKey,
  MlKem1024EncapsulationKey as EncapsulationKey, MlKem1024Seed as Seed, MlKemError,
};

include!("common/mlkem.rs");
