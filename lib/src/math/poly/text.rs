//! The text of a polynomial.

use super::Polynomial;
use super::coeff::Coefficient;
use core::fmt;

impl<T: Coefficient> fmt::Display for Polynomial<T> {
    /// Writes the terms from the leading one down, as PARI/GP does: `3*x^2 - x + 1`, `0` for the
    /// zero polynomial; each coefficient as its type's `Display` writes it, with the formatter's
    /// options, and a complex one in parentheses: `(1+2i)*x`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.c.is_empty() {
            return f.write_str("0");
        }
        let mut first = true;
        for (i, c) in self.c.iter().enumerate().rev() {
            if c.is_zero() {
                continue;
            }
            let negative = c.is_negative();
            match (first, negative) {
                (true, true) => f.write_str("-")?,
                (true, false) => {}
                (false, true) => f.write_str(" - ")?,
                (false, false) => f.write_str(" + ")?,
            }
            first = false;
            let unit = c.is_one() || (negative && T::neg(c).is_one());
            if i == 0 || !unit {
                c.write(f, true)?;
                if i > 0 {
                    f.write_str("*")?;
                }
            }
            match i {
                0 => {}
                1 => f.write_str("x")?,
                _ => write!(f, "x^{i}")?,
            }
        }
        Ok(())
    }
}
