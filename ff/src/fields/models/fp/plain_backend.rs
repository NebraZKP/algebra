///! A (WIP) backend using plain (non-Montgomery) representation, for platforms
///! with accelerated operations for this format.
///!
///! Notably, the risc0 ZKVM supports a `bigint` syscall which performs mulmod
///! of 256 bit numbers.
///!
use crate::{BigInt, BigInteger, Field, Fp, Fp256, FpConfig, PrimeField, SqrtPrecomputation, Zero};
use core::marker::PhantomData;

#[cfg(target_vendor = "risc0")]
extern "C" {
    /// The risc0 syscall for mulmod
    fn sys_bigint(
        result: *mut [u32; 8],
        op: u32,
        x: *const [u32; 8],
        y: *const [u32; 8],
        modulus: *const [u32; 8],
    );
}

#[inline(always)]
fn __sub_with_borrow(a: &mut BigInt<4>, b: &BigInt<4>) -> bool {
    use crate::biginteger::arithmetic::sbb_for_sub_with_borrow as sbb;
    let mut borrow = 0;
    borrow = sbb(&mut a.0[0], b.0[0], borrow);
    borrow = sbb(&mut a.0[1], b.0[1], borrow);
    borrow = sbb(&mut a.0[2], b.0[2], borrow);
    borrow = sbb(&mut a.0[3], b.0[3], borrow);
    borrow != 0
}

#[inline(always)]
fn __add_with_carry(a: &mut BigInt<4>, b: &BigInt<4>) -> bool {
    use crate::biginteger::arithmetic::adc_for_add_with_carry as adc;
    let mut carry = 0;
    carry = adc(&mut a.0[0], b.0[0], carry);
    carry = adc(&mut a.0[1], b.0[1], carry);
    carry = adc(&mut a.0[2], b.0[2], carry);
    carry = adc(&mut a.0[3], b.0[3], carry);
    carry != 0
}

/// Per-field configuration. Basic properties of the field are given by
/// implementing this trait.  It provides some trivial bigint operations (mostly
/// unrolled from the `montgomery_backend` implementation and macros) for use by
/// the full `Fp256PlainBackend<Self>` which fully implements `FpConfig<4>`.
pub trait Fp256PlainConfig: 'static + Sync + Send + Sized {
    const MODULUS: BigInt<4>;
    const GENERATOR: Fp256<Fp256PlainBackend<Self>>;
    const ZERO: Fp256<Fp256PlainBackend<Self>>;
    const ONE: Fp256<Fp256PlainBackend<Self>>;
    const TWO_ADICITY: u32;
    const TWO_ADIC_ROOT_OF_UNITY: Fp256<Fp256PlainBackend<Self>>;

    #[cfg(not(target_vendor = "risc0"))]
    /// Only required on host-side
    type FullImplConfig: FpConfig<4>;

    #[inline(always)]
    fn __subtract_modulus(a: &mut Fp256<Fp256PlainBackend<Self>>) {
        if a.is_geq_modulus() {
            __sub_with_borrow(&mut a.0, &Self::MODULUS);
        }
    }

    #[inline(always)]
    fn __subtract_modulus_with_carry(a: &mut Fp256<Fp256PlainBackend<Self>>, carry: bool) {
        if a.is_geq_modulus() || carry {
            __sub_with_borrow(&mut a.0, &Self::MODULUS);
        }
    }
}

/// Provides a full implementation of `FpConfig<4>`, using a PlainFp256Config.
///
/// Note, this could be done in a single implementation.  The split here between
/// Fp256PlainBackend and Fp256PlainConfig reflects the original Montgomery
/// version upon which this is based.
pub struct Fp256PlainBackend<T: Fp256PlainConfig>(PhantomData<T>);

impl<T: Fp256PlainConfig> FpConfig<4> for Fp256PlainBackend<T> {
    const MODULUS: BigInt<4> = T::MODULUS;
    const GENERATOR: Fp256<Self> = T::GENERATOR;
    const ZERO: Fp256<Self> = T::ZERO;
    const ONE: Fp256<Self> = T::ONE;
    const TWO_ADICITY: u32 = T::TWO_ADICITY;
    const TWO_ADIC_ROOT_OF_UNITY: Fp256<Self> = T::TWO_ADIC_ROOT_OF_UNITY;
    const SQRT_PRECOMP: Option<SqrtPrecomputation<Fp256<Self>>> = None;

    fn add_assign(a: &mut Fp256<Self>, b: &Fp256<Self>) {
        __add_with_carry(&mut a.0, &b.0);
        T::__subtract_modulus(a);
    }

    fn sub_assign(a: &mut Fp256<Self>, b: &Fp256<Self>) {
        // If `other` is larger than `self`, add the modulus to self first.
        if b.0 > a.0 {
            __add_with_carry(&mut a.0, &Self::MODULUS);
        }
        __sub_with_borrow(&mut a.0, &b.0);
    }

    fn double_in_place(a: &mut Fp256<Self>) {
        let b = a.0;
        __add_with_carry(&mut a.0, &b);
        T::__subtract_modulus(a);
    }

    fn neg_in_place(a: &mut Fp256<Self>) {
        if *a != Fp::<Self, 4>::ZERO {
            let mut tmp = Self::MODULUS;
            __sub_with_borrow(&mut tmp, &a.0);
            a.0 = tmp;
        }
    }

    fn mul_assign(a: &mut Fp256<Self>, b: &Fp256<Self>) {
        #[cfg(target_vendor = "risc0")]
        {
            let a_copy = a.clone();
            #[allow(unsafe_code)]
            unsafe {
                let mod_ptr: *const [u32; 8] =
                    (&Self::MODULUS.0) as *const [u64; 4] as *const [u32; 8];
                let a_ptr: *const [u32; 8] = (&a_copy.0 .0) as *const [u64; 4] as *const [u32; 8];
                let b_ptr: *const [u32; 8] = (&b.0 .0) as *const [u64; 4] as *const [u32; 8];
                let out_ptr: *mut [u32; 8] = (&mut a.0 .0) as *mut [u64; 4] as *mut [u32; 8];
                sys_bigint(out_ptr, 0, a_ptr, b_ptr, mod_ptr);
            }
        }
        #[cfg(not(target_vendor = "risc0"))]
        {
            // Naive implementation for now
            let mut aa = Fp256::<T::FullImplConfig>::from_bigint(a.0).unwrap();
            let bb = Fp256::<T::FullImplConfig>::from_bigint(b.0).unwrap();
            aa *= bb;

            a.0 = aa.into_bigint();
        }
    }

    fn sum_of_products<const N: usize>(a: &[Fp256<Self>; N], b: &[Fp256<Self>; N]) -> Fp256<Self> {
        // Naive implementation
        if N == 0 {
            Fp256::<Self>::ZERO
        } else {
            let mut sum = a[0] * b[0];
            for i in 1..N {
                sum += a[i] * b[i];
            }

            sum
        }
    }

    fn square_in_place(a: &mut Fp256<Self>) {
        // Naive implementation for now
        let b = a.clone();
        Self::mul_assign(a, &b)
    }
    fn inverse(a: &Fp256<Self>) -> Option<Fp256<Self>> {
        if a.is_zero() {
            None
        } else {
            // Guajardo Kumar Paar Pelzl
            // Efficient Software-Implementation of Finite Fields with Applications to
            // Cryptography
            // Algorithm 16 (BEA for Inversion in Fp)

            let one = BigInt::<4>::from(1u64);

            let mut u = a.0;
            let mut v = Self::MODULUS;
            let mut b = Fp::ONE;
            let mut c = Fp::ZERO;

            while u != one && v != one {
                while u.is_even() {
                    u.div2();

                    if b.0.is_even() {
                        b.0.div2();
                    } else {
                        b.0.add_with_carry(&Self::MODULUS);
                        b.0.div2();
                    }
                }

                while v.is_even() {
                    v.div2();

                    if c.0.is_even() {
                        c.0.div2();
                    } else {
                        c.0.add_with_carry(&Self::MODULUS);
                        c.0.div2();
                    }
                }

                if v < u {
                    u.sub_with_borrow(&v);
                    b -= &c;
                } else {
                    v.sub_with_borrow(&u);
                    c -= &b;
                }
            }

            if u == one {
                Some(b)
            } else {
                Some(c)
            }
        }
    }

    fn from_bigint(v: BigInt<4>) -> Option<Fp256<Self>> {
        let v = Fp::<Self, 4>(v, PhantomData);
        if v.is_geq_modulus() {
            None
        } else {
            Some(v)
        }
    }

    fn into_bigint(v: Fp256<Self>) -> BigInt<4> {
        v.0
    }
}

impl<T: Fp256PlainConfig> Fp256<Fp256PlainBackend<T>> {
    // const fn const_is_zero(&self) -> bool {
    //     self.0.const_is_zero()
    // }

    // const fn const_neg(self) -> Self {
    //     if !self.const_is_zero() {
    //         let mut out = T::MODULUS;
    //         __sub_with_borrow(&mut out, &self.0);
    //         Self(out, PhantomData)
    //     } else {
    //         self
    //     }
    // }

    #[doc(hidden)]
    pub const fn plain_from_sign_and_limbs(is_positive: bool, limbs: &[u64]) -> Self {
        let mut repr = BigInt([0; 4]);
        assert!(limbs.len() <= 4);
        crate::const_for!((i in 0..(limbs.len())) {
            repr.0[i] = limbs[i];
        });
        // TODO: assert is less than modulus
        // assert!(repr <= T::MODULUS);
        assert!(is_positive);
        Self(repr, PhantomData)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // TODO: expand these tests.
    // Test against the Montgomery backend for the bn254::Fq field

    pub struct FqPlainConfig;

    impl Fp256PlainConfig for FqPlainConfig {
        // 21888242871839275222246405745257275088696311157297823662689037894645226208583
        const MODULUS: BigInt<4> = BigInt([
            4332616871279656263,
            10917124144477883021,
            13281191951274694749,
            3486998266802970665,
        ]);
        const GENERATOR: Fp256<Fp256PlainBackend<Self>> = Fp(BigInt([3, 0, 0, 0]), PhantomData);
        const ZERO: Fp256<Fp256PlainBackend<Self>> = Fp(BigInt([0, 0, 0, 0]), PhantomData);
        const ONE: Fp256<Fp256PlainBackend<Self>> = Fp(BigInt([1, 0, 0, 0]), PhantomData);
        const TWO_ADICITY: u32 = 2;
        // 21888242871839275222246405745257275088696311157297823662689037894645226208582
        const TWO_ADIC_ROOT_OF_UNITY: Fp256<Fp256PlainBackend<Self>> = Fp(
            BigInt([
                4332616871279656262,
                10917124144477883021,
                13281191951274694749,
                3486998266802970665,
            ]),
            PhantomData,
        );

        type FullImplConfig = Fp256PlainBackend<Self>; // MontBackend<ark_bn254::fq::FqConfig, 4>;
                                                       // type FullImplConfig = MontBackend<ark_bn254::fq::FqConfig, 4>;
    }

    type Fq = Fp256<Fp256PlainBackend<FqPlainConfig>>;

    #[test]
    fn test_bn256_fq() {
        // sage: x = Fq(-13)
        // sage: y = Fq.random_element()

        // sage: u256_to_u64s(int(x))
        let x = Fq::from_bigint(BigInt::new([
            4332616871279656250,
            10917124144477883021,
            13281191951274694749,
            3486998266802970665,
        ]))
        .unwrap();

        // sage: u256_to_u64s(int(y))
        let y = Fq::from_bigint(BigInt::new([
            14149095143606734407,
            15470625807535465084,
            3942277989892989553,
            1637495180493918181,
        ]))
        .unwrap();

        // sage: u256_to_u64s(int(x + y))
        let expect_x_plus_y = Fq::from_bigint(BigInt::new([
            14149095143606734394,
            15470625807535465084,
            3942277989892989553,
            1637495180493918181,
        ]))
        .unwrap();

        // sage: u256_to_u64s(int(x * y))
        let expect_x_times_y = Fq::from_bigint(BigInt::new([
            12410777895456011094,
            4428942029350996358,
            4825241642894895815,
            3121550521199858304,
        ]))
        .unwrap();

        // sage: u256_to_u64s(int(x - y))
        let expect_x_minus_y = Fq::from_bigint(BigInt::new([
            8630265801382473459,
            13893242410651969552,
            9338913961381705195,
            1849503086309052484,
        ]))
        .unwrap();

        // sage: u256_to_u64s(int(Fq(1) / y))
        let expect_y_inverse = Fq::from_bigint(BigInt::new([
            12131777808620374806,
            14403703374183863780,
            13988826170433030724,
            396543726139761109,
        ]))
        .unwrap();

        assert_eq!(expect_x_plus_y, x + y);

        {
            let mut x_plus_y = x;
            x_plus_y += y;
            assert_eq!(expect_x_plus_y, x_plus_y);
        }

        assert_eq!(expect_x_minus_y, x - y);

        {
            let mut x_minus_y = x;
            x_minus_y -= y;
            assert_eq!(expect_x_minus_y, x_minus_y);
        }

        {
            let mut y_minus_x = y;
            y_minus_x -= x;
            assert_eq!(-expect_x_minus_y, y_minus_x);
        }

        // {
        //     let x_times_y = x * y;
        //     assert_eq!(expect_x_times_y, x_times_y);
        // }

        // {
        //     let minus_y = -y;
        //     let inv = minus_y.inverse().unwrap();
        //     assert_eq!(expect_inv, inv);

        //     assert_eq!(Fq::from(-1).inverse(), Some(Fq::from(-1)));
        // }

        // {
        //     let mut x = x;
        //     x.square_in_place();
        //     assert_eq!(expect_x_squared, x);
        // }
    }
}
