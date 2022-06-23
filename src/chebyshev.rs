/// A linear combinations of Chebyshev's polynomials of first kind, defined on an closed interval
/// of ℝ
#[derive(Debug, Clone)]
pub struct ChebyshevPolynomial {
    /// The coefficients of the linear combination.
    ///
    /// `self` represents the polynomials `\sum_{i < coeffs.len()} coeffs[i] T_i` where `T_i` is
    /// the `i-th` Chebyshev's polynomial of first kind.
    ///
    /// If coeffs is empty, `self` represents the null polynomial.
    pub coeffs: Vec<f64>,
    pub(crate) definition_interval: [f64; 2],
}

impl ChebyshevPolynomial {
    /// Evaluate `self` at `t`.
    pub fn evaluate(&self, t: f64) -> f64 {
        if self.coeffs.len() == 0 {
            0.
        } else if self.coeffs.len() == 1 {
            self.coeffs[0]
        } else {
            let a = self.definition_interval[0];
            let b = self.definition_interval[1];
            let u = (2. * t - (a + b)) / (b - a);

            #[allow(non_snake_case)]
            let mut T_previous = 1.;
            #[allow(non_snake_case)]
            let mut T = u;
            let mut v = self.coeffs[0] + u * self.coeffs[1];
            for coeff in self.coeffs.iter().skip(2) {
                #[allow(non_snake_case)]
                let T_next = 2. * u * T - T_previous;
                T_previous = T;
                T = T_next;
                v += T * coeff;
            }
            v
        }
    }

    /// Return the interval on which `self` is defined.
    pub fn definition_interval(&self) -> [f64; 2] {
        self.definition_interval
    }

    pub fn from_coeffs_interval(coeffs: Vec<f64>, definition_interval: [f64; 2]) -> Self {
        Self {
            coeffs,
            definition_interval,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ChebyshevPolynomial;

    #[test]
    /// Check that the equation `T_n(cos theta) = cos(n theta)` is verrified
    fn trigonometric_property() {
        for n in 0..10 {
            let mut polynomial = ChebyshevPolynomial {
                coeffs: vec![0.; n + 1],
                definition_interval: [-1., 1.],
            };
            polynomial.coeffs[n] = 1.;
            let theta = 1.234;

            let expected = (n as f64 * theta).cos();
            let result = polynomial.evaluate(theta.cos());
            assert!((expected - result).abs() < 1e-5, "Failed for n = {n}");
        }
    }
}
