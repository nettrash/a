//! `derivative`, `second_derivative`, `gradient`, `jacobian`, `hessian` and their `value_and_…`
//! forms: exact cases, a function of three variables against SymPy and mpmath, Jacobians of known
//! maps, the symmetry of Hessians by independent passes, the number of passes, empty points and
//! the panics on wrong shapes.

use a::math::dual::{
    Dual, DualNum, HyperDual, derivative, gradient, hessian, jacobian, second_derivative,
    value_and_derivative, value_and_derivatives, value_and_gradient, value_and_jacobian,
    value_gradient_and_hessian,
};
use a::math::matrix::Matrix;
use core::cell::Cell;

const DATA: &str = include_str!("data/reference.txt");

fn column(v: &[f64]) -> Matrix<f64> {
    Matrix::from_column_major(v.len(), 1, v.to_vec())
}

/// The function of three variables of the reference data.
fn three<D: DualNum<Float = f64>>(v: &[D]) -> D {
    let (x, y, z) = (v[0], v[1], v[2]);
    (x * y).exp() * z.sin() + (x * x + y * y + 1.0).ln() * z + (x - z * z).atan()
}

#[test]
fn one_variable() {
    fn f<D: DualNum<Float = f64>>(x: D) -> D {
        x.powi(3) - x * 2.0
    }
    assert_eq!(derivative(f, 2.0), 10.0);
    assert_eq!(value_and_derivative(f, 2.0), (4.0, 10.0));
    assert_eq!(second_derivative(f, 2.0), 12.0);
    assert_eq!(value_and_derivatives(f, 2.0), (4.0, 10.0, 12.0));
    assert_eq!(derivative(f, 2.0_f32 as f64), 10.0);
    assert_eq!(derivative(|x: Dual<f32>| x.sin(), 0.0_f32), 1.0);
    // Through every nesting: the third derivative of x⁴ is 24x.
    let d3 = derivative(
        |x: Dual<f64>| second_derivative(|y: HyperDual<Dual<f64>>| y.powi(4), x),
        1.5,
    );
    assert_eq!(d3, 36.0);
    let d3b = second_derivative(
        |x: HyperDual<f64>| derivative(|y: Dual<HyperDual<f64>>| y.powi(4), x),
        1.5,
    );
    assert_eq!(d3b, 36.0);
    // One evaluation a pass.
    let calls = Cell::new(0);
    let _ = value_and_derivatives(
        |x: HyperDual<f64>| {
            calls.set(calls.get() + 1);
            x
        },
        1.0,
    );
    assert_eq!(calls.get(), 1);
}

#[test]
fn gradient_and_hessian_of_three_variables_match_mpmath() {
    let mut points = 0;
    for line in DATA.lines().filter(|l| l.starts_with("g ")) {
        let w: Vec<f64> = line
            .split_whitespace()
            .skip(3)
            .map(|s| s.parse().unwrap())
            .collect();
        let (p, want) = (&w[..3], &w[3..]);
        let x = column(p);
        let (v, g, h) = value_gradient_and_hessian(three, &x);
        let close = |got: f64, want: f64, ulps: f64| {
            (got - want).abs() <= ulps * f64::EPSILON * want.abs().max(1.0)
        };
        assert!(close(v, want[0], 4.0), "{line}");
        assert_eq!(three(x.as_slice()), v);
        for i in 0..3 {
            assert!(
                close(g[(i, 0)], want[1 + i], 8.0),
                "{line}: ∂{i} {}",
                g[(i, 0)]
            );
        }
        let order = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
        for (k, &(i, j)) in order.iter().enumerate() {
            assert!(
                close(h[(i, j)], want[4 + k], 16.0),
                "{line}: ∂{i}∂{j} {}",
                h[(i, j)]
            );
            assert_eq!(h[(i, j)], h[(j, i)]);
        }
        assert_eq!(gradient(three, &x), g);
        assert_eq!(value_and_gradient(three, &x), (v, g.clone()));
        assert_eq!(hessian(three, &x), h);
        // f32 to its precision.
        let x32 = Matrix::from_column_major(3, 1, p.iter().map(|&t| t as f32).collect());
        fn three32<D: DualNum<Float = f32>>(v: &[D]) -> D {
            let (x, y, z) = (v[0], v[1], v[2]);
            (x * y).exp() * z.sin() + (x * x + y * y + 1.0).ln() * z + (x - z * z).atan()
        }
        let h32 = hessian(three32, &x32);
        for (k, &(i, j)) in order.iter().enumerate() {
            let e = (f64::from(h32[(i, j)]) - want[4 + k]).abs();
            assert!(
                e <= 64.0 * f64::from(f32::EPSILON) * want[4 + k].abs().max(1.0),
                "{line}"
            );
        }
        points += 1;
    }
    assert_eq!(points, 4);
}

#[test]
fn hessians_are_symmetric_by_independent_passes() {
    // Each entry by the pass with ε₁ on xᵢ and ε₂ on xⱼ, and again with the two swapped, and as
    // ∂/∂xᵢ of ∂f/∂xⱼ by nested dual numbers: three independent sums of rounded terms.
    let x = column(&[0.7, -1.3, 0.4]);
    let h = hessian(three, &x);
    for i in 0..3 {
        for j in 0..3 {
            let swapped: Vec<HyperDual<f64>> = (0..3)
                .map(|k| {
                    let e1 = if k == j { 1.0 } else { 0.0 };
                    let e2 = if k == i { 1.0 } else { 0.0 };
                    HyperDual::new(x[(k, 0)], e1, e2, 0.0)
                })
                .collect();
            let s = three(&swapped).eps1eps2;
            let nested: Vec<Dual<Dual<f64>>> = (0..3)
                .map(|k| {
                    let inner = if k == i { 1.0 } else { 0.0 };
                    let outer = if k == j { 1.0 } else { 0.0 };
                    Dual::new(Dual::new(x[(k, 0)], inner), Dual::new(outer, 0.0))
                })
                .collect();
            let n = three(&nested).eps.eps;
            let tol = 8.0 * f64::EPSILON * h[(i, j)].abs().max(1.0);
            assert!((s - h[(i, j)]).abs() <= tol, "{i} {j}: {s} {}", h[(i, j)]);
            assert!((n - h[(i, j)]).abs() <= tol, "{i} {j}: {n} {}", h[(i, j)]);
        }
    }
}

#[test]
fn derivatives_of_quadratic_forms_are_exact() {
    // f(x) = ½xᵀAx + bᵀx with A symmetric: ∇f = Ax + b, H = A, in small integers, exactly.
    let a = Matrix::from_rows(&[[4.0, 1.0, -2.0], [1.0, 3.0, 0.0], [-2.0, 0.0, 5.0]]);
    let b = [1.0, -1.0, 2.0];
    let f = |v: &[HyperDual<f64>]| {
        let mut s = HyperDual::constant(0.0);
        for i in 0..3 {
            for j in 0..3 {
                s += v[i] * v[j] * (0.5 * a[(i, j)]);
            }
            s += v[i] * b[i];
        }
        s
    };
    let x = column(&[1.0, 2.0, -1.0]);
    let (_, g, h) = value_gradient_and_hessian(f, &x);
    assert_eq!(h, a);
    assert_eq!(g, &(&a * &x) + &column(&b));
}

#[test]
fn jacobians_of_known_maps() {
    // Polar to Cartesian coordinates: [[cos θ, −r sin θ], [sin θ, r cos θ]].
    let polar = |p: &[Dual<f64>]| [p[0] * p[1].cos(), p[0] * p[1].sin()];
    let (r, t) = (2.0, 0.7_f64);
    let (v, j) = value_and_jacobian(polar, &column(&[r, t]));
    assert_eq!(v.as_slice(), &[r * t.cos(), r * t.sin()]);
    assert_eq!(
        j,
        Matrix::from_rows(&[[t.cos(), -r * t.sin()], [t.sin(), r * t.cos()]])
    );
    // Spherical to Cartesian: its determinant is r² sin θ.
    let sph = |p: &[Dual<f64>]| {
        let (r, th, ph) = (p[0], p[1], p[2]);
        vec![
            r * th.sin() * ph.cos(),
            r * th.sin() * ph.sin(),
            r * th.cos(),
        ]
    };
    let (r, th, ph) = (1.5, 0.9_f64, -0.4_f64);
    let j = jacobian(sph, &column(&[r, th, ph]));
    let det = j.det();
    assert!((det - r * r * th.sin()).abs() < 1e-15);
    // A linear map is its own Jacobian, exactly, whatever the point; and m ≠ n.
    let a = Matrix::from_rows(&[[1.0, -2.0], [0.5, 3.0], [4.0, 0.0]]);
    let lin = |p: &[Dual<f64>]| {
        (0..3)
            .map(|i| p[0] * a[(i, 0)] + p[1] * a[(i, 1)])
            .collect::<Vec<_>>()
    };
    let (v, j) = value_and_jacobian(lin, &column(&[7.0, -3.0]));
    assert_eq!(j, a);
    assert_eq!(v, &a * &column(&[7.0, -3.0]));
    // The Jacobian of a gradient is the Hessian, as a check across the two.
    let x = column(&[0.3, 0.8, -0.6]);
    let jg = jacobian(
        |v: &[Dual<f64>]| {
            let p: Vec<Dual<Dual<f64>>> = (0..3)
                .map(|k| Dual::new(v[k], Dual::new(0.0, 0.0)))
                .collect();
            (0..3)
                .map(|i| {
                    let mut q = p.clone();
                    q[i].eps = Dual::new(1.0, 0.0);
                    three(&q).eps
                })
                .collect::<Vec<_>>()
        },
        &x,
    );
    let h = hessian(three, &x);
    for i in 0..3 {
        for k in 0..3 {
            assert!((jg[(i, k)] - h[(i, k)]).abs() <= 1e-14 * h[(i, k)].abs().max(1.0));
        }
    }
}

#[test]
fn passes_are_counted_as_stated() {
    let calls = Cell::new(0usize);
    let f = |v: &[Dual<f64>]| {
        calls.set(calls.get() + 1);
        v.iter().copied().sum::<Dual<f64>>()
    };
    let x = column(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(gradient(f, &x).as_slice(), &[1.0; 5]);
    assert_eq!(calls.replace(0), 5);
    let fj = |v: &[Dual<f64>]| {
        calls.set(calls.get() + 1);
        vec![v[0] * v[1], v[2] + v[3] + v[4]]
    };
    let j = jacobian(fj, &x);
    assert_eq!(j.shape(), (2, 5));
    assert_eq!(calls.replace(0), 5);
    let fh = |v: &[HyperDual<f64>]| {
        calls.set(calls.get() + 1);
        v[0] * v[1] * v[4]
    };
    let h = hessian(fh, &x);
    assert_eq!(calls.replace(0), 15);
    assert_eq!(
        (h[(0, 1)], h[(0, 4)], h[(1, 4)], h[(0, 0)], h[(2, 3)]),
        (5.0, 2.0, 1.0, 0.0, 0.0)
    );
}

#[test]
fn empty_points() {
    let calls = Cell::new(0usize);
    let f = |_: &[Dual<f64>]| {
        calls.set(calls.get() + 1);
        Dual::new(7.0, 0.0)
    };
    let e = Matrix::from_column_major(0, 1, Vec::new());
    assert_eq!(gradient(f, &e).shape(), (0, 1));
    assert_eq!(calls.get(), 0);
    let (v, g) = value_and_gradient(f, &e);
    assert_eq!((v, g.shape(), calls.get()), (7.0, (0, 1), 1));
    let (v, j) = value_and_jacobian(
        |_: &[Dual<f64>]| [Dual::new(1.0, 0.0), Dual::new(2.0, 0.0)],
        &e,
    );
    assert_eq!((v.as_slice(), j.shape()), (&[1.0, 2.0][..], (2, 0)));
    let fh = |_: &[HyperDual<f64>]| HyperDual::new(3.0, 0.0, 0.0, 0.0);
    assert_eq!(hessian(fh, &e).shape(), (0, 0));
    let (v, g, h) = value_gradient_and_hessian(fh, &e);
    assert_eq!((v, g.shape(), h.shape()), (3.0, (0, 1), (0, 0)));
    // One variable: the free functions of one variable and of several agree.
    let one = column(&[0.6]);
    let s = |v: &[HyperDual<f64>]| v[0].sin() * v[0];
    let d = value_and_derivatives(|x: HyperDual<f64>| x.sin() * x, 0.6);
    let (v, g, h) = value_gradient_and_hessian(s, &one);
    assert_eq!((v, g[(0, 0)], h[(0, 0)]), d);
}

#[test]
#[should_panic(expected = "the point is 1×2, not a column")]
fn gradient_of_a_row_panics() {
    gradient(|v: &[Dual<f64>]| v[0], &Matrix::from_rows(&[[1.0, 2.0]]));
}

#[test]
#[should_panic(expected = "the point is 2×2, not a column")]
fn hessian_of_a_square_panics() {
    hessian(
        |v: &[HyperDual<f64>]| v[0],
        &Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]]),
    );
}

#[test]
#[should_panic(expected = "the point is 0×0, not a column")]
fn jacobian_of_an_empty_matrix_panics() {
    jacobian(
        |v: &[Dual<f64>]| v.to_vec(),
        &Matrix::from_column_major(0, 0, Vec::new()),
    );
}

#[test]
#[should_panic(expected = "the function gave 1 values in pass 1, and 2 in the first")]
fn jacobian_of_a_function_of_changing_length_panics() {
    let calls = Cell::new(0);
    jacobian(
        |v: &[Dual<f64>]| {
            calls.set(calls.get() + 1);
            vec![v[0]; if calls.get() == 1 { 2 } else { 1 }]
        },
        &column(&[1.0, 2.0]),
    );
}
