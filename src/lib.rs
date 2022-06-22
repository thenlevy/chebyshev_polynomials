pub(crate) mod chebyshev;
pub use chebyshev::*;
pub mod interpolation;

pub use interpolation::*;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
