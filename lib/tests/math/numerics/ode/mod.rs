//! The initial value problem: the coefficients of the explicit methods, the methods against
//! Hairer's codes, SciPy and the exact solutions, the dense output, the events, the step-size
//! control and the failures of the driver, and the stiff methods on the same interface.

mod bdf;
mod dense;
mod driver;
mod events;
mod explicit;
mod problems;
mod radau;
mod reference;
mod stiff;
mod stiff_problems;
mod stiff_reference;
mod tableau;
