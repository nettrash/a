//! a::math::quaternion's Quaternion<f32/f64> against glam (Quat with NEON/SSE for f32, DQuat for
//! f64) and nalgebra (Quaternion, UnitQuaternion), per operation over a pool of 1024 random
//! inputs: unit quaternions for the rotations (glam and nalgebra's UnitQuaternion assume them;
//! ours takes any norm and reads q/|q|), random ones for the algebra. Every side's results are
//! compared with ours (to a tolerance, and up to the sign of q where a rotation's quaternion is
//! the result) before timing.

#[path = "../common.rs"]
mod common;

use a::math::matrix::Matrix;
use a::math::quaternion::{EulerFrame, EulerSequence, Quaternion};
use common::*;
use glam::EulerRot;


/// The largest difference between ours and a reference over the pool, scaled; panics beyond tol.
fn agree(what: &str, name: &str, ours: &[Vec<f64>], theirs: &[Vec<f64>], tol: f64, up_to_sign: bool) -> f64 {
    let mut worst = 0f64;
    for (i, (a, b)) in ours.iter().zip(theirs).enumerate() {
        let d = |s: f64| a.iter().zip(b).map(|(x, y)| (x - s * y).abs() / (1.0 + x.abs())).fold(0f64, f64::max);
        let e = if up_to_sign { d(1.0).min(d(-1.0)) } else { d(1.0) };
        if !(e <= tol) {
            panic!("{what}: {name} differs from ours at item {i}: {a:?} vs {b:?}");
        }
        worst = worst.max(e);
    }
    worst
}

macro_rules! quat_bench {
    ($fname:ident, $t:ty, $tn:literal, $gq:ty, $gv:ty, $gm:ty, $tol:expr) => {
        #[allow(clippy::too_many_lines)]
        fn $fname(out: &Out, seed: u64) {
            type T = $t;
            type Q = Quaternion<T>;
            type GQ = $gq;
            type GV = $gv;
            type GM = $gm;
            type NQ = nalgebra::Quaternion<T>;
            type NU = nalgebra::UnitQuaternion<T>;
            type NV = nalgebra::Vector3<T>;
            let tol: f64 = $tol;
            let g = "quaternion";
            let p = |s: &str| format!("{}, {s}", $tn);
            let len = 1024usize;
            let mut rng = Rng::new(seed);
            let rq = |rng: &mut Rng| -> [f64; 4] { [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)] };
            let unit = |rng: &mut Rng| -> [f64; 4] {
                let q = rq(rng);
                let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
                [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
            };
            let rv = |rng: &mut Rng| -> [f64; 3] { [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)] };
            let o = |q: &[f64; 4]| Q::new(q[0] as T, q[1] as T, q[2] as T, q[3] as T);
            let gq = |q: &[f64; 4]| GQ::from_xyzw(q[1] as T, q[2] as T, q[3] as T, q[0] as T);
            let nq = |q: &[f64; 4]| NQ::new(q[0] as T, q[1] as T, q[2] as T, q[3] as T);
            let nu = |q: &[f64; 4]| NU::new_unchecked(nq(q));
            let ov = |v: &[f64; 3]| [v[0] as T, v[1] as T, v[2] as T];
            let gv = |v: &[f64; 3]| GV::new(v[0] as T, v[1] as T, v[2] as T);
            let nv = |v: &[f64; 3]| NV::new(v[0] as T, v[1] as T, v[2] as T);
            let qo = |q: &Q| vec![q.w as f64, q.x as f64, q.y as f64, q.z as f64];
            let qg = |q: &GQ| vec![q.w as f64, q.x as f64, q.y as f64, q.z as f64];
            let qn = |q: &NQ| vec![q.w as f64, q.i as f64, q.j as f64, q.k as f64];
            let vo = |v: &[T; 3]| v.iter().map(|x| *x as f64).collect::<Vec<f64>>();
            let vg = |v: &GV| vec![v.x as f64, v.y as f64, v.z as f64];
            let vn = |v: &NV| vec![v.x as f64, v.y as f64, v.z as f64];

            // pools
            let qa: Vec<[f64; 4]> = (0..len).map(|_| rq(&mut rng)).collect();
            let qb: Vec<[f64; 4]> = (0..len).map(|_| rq(&mut rng)).collect();
            let ua: Vec<[f64; 4]> = (0..len).map(|_| unit(&mut rng)).collect();
            let ub: Vec<[f64; 4]> = (0..len).map(|_| unit(&mut rng)).collect();
            let vs: Vec<[f64; 3]> = (0..len).map(|_| rv(&mut rng)).collect();
            let ts: Vec<f64> = (0..len).map(|_| rng.f64()).collect();

            let o2: Vec<(Q, Q)> = qa.iter().zip(&qb).map(|(a, b)| (o(a), o(b))).collect();
            let g2: Vec<(GQ, GQ)> = qa.iter().zip(&qb).map(|(a, b)| (gq(a), gq(b))).collect();
            let n2: Vec<(NQ, NQ)> = qa.iter().zip(&qb).map(|(a, b)| (nq(a), nq(b))).collect();
            let ou: Vec<(Q, Q, T)> = ua.iter().zip(&ub).zip(&ts).map(|((a, b), t)| (o(a), o(b), *t as T)).collect();
            let gu: Vec<(GQ, GQ, T)> = ua.iter().zip(&ub).zip(&ts).map(|((a, b), t)| (gq(a), gq(b), *t as T)).collect();
            let nuu: Vec<(NU, NU, T)> = ua.iter().zip(&ub).zip(&ts).map(|((a, b), t)| (nu(a), nu(b), *t as T)).collect();
            let ouv: Vec<(Q, [T; 3])> = ua.iter().zip(&vs).map(|(a, v)| (o(a), ov(v))).collect();
            let guv: Vec<(GQ, GV)> = ua.iter().zip(&vs).map(|(a, v)| (gq(a), gv(v))).collect();
            let nuv: Vec<(NU, NV)> = ua.iter().zip(&vs).map(|(a, v)| (nu(a), nv(v))).collect();

            macro_rules! row {
                ($op:expr, $params:expr, $notes:expr, $sides:expr) => {
                    out.row(g, $op, &$params, len, $notes, $sides)
                };
            }

            // ---- the algebra ------------------------------------------------------------
            {
                let w: Vec<Vec<f64>> = o2.iter().map(|(a, b)| qo(&(*a * *b))).collect();
                let e1 = agree("mul", "glam", &w, &g2.iter().map(|(a, b)| qg(&(*a * *b))).collect::<Vec<_>>(), tol, false);
                let e2 = agree("mul", "nalgebra", &w, &n2.iter().map(|(a, b)| qn(&(a * b))).collect::<Vec<_>>(), tol, false);
                let _ = (e1, e2);
                row!("mul", p("random quaternions"), "", vec![
                    mk("ours", "", &o2, |(a, b)| *a * *b),
                    mk("glam", "", &g2, |(a, b)| *a * *b),
                    mk("nalgebra", "Quaternion", &n2, |(a, b)| a * b),
                ]);

                let w: Vec<Vec<f64>> = o2.iter().map(|(a, _)| qo(&a.normalize())).collect();
                agree("normalize", "glam", &w, &g2.iter().map(|(a, _)| qg(&a.normalize())).collect::<Vec<_>>(), tol, false);
                agree("normalize", "nalgebra", &w, &n2.iter().map(|(a, _)| qn(&a.normalize())).collect::<Vec<_>>(), tol, false);
                row!("normalize", p("random quaternions"), "", vec![
                    mk("ours", "scaled where the squares would overflow", &o2, |(a, _)| a.normalize()),
                    mk("glam", "", &g2, |(a, _)| a.normalize()),
                    mk("nalgebra", "Quaternion::normalize", &n2, |(a, _)| a.normalize()),
                ]);

                let w: Vec<Vec<f64>> = o2.iter().map(|(a, _)| vec![a.norm() as f64]).collect();
                agree("norm", "glam", &w, &g2.iter().map(|(a, _)| vec![a.length() as f64]).collect::<Vec<_>>(), tol, false);
                agree("norm", "nalgebra", &w, &n2.iter().map(|(a, _)| vec![a.norm() as f64]).collect::<Vec<_>>(), tol, false);
                row!("norm", p("random quaternions"), "", vec![
                    mk("ours", "", &o2, |(a, _)| a.norm()),
                    mk("glam", "length", &g2, |(a, _)| a.length()),
                    mk("nalgebra", "", &n2, |(a, _)| a.norm()),
                ]);

                let w: Vec<Vec<f64>> = o2.iter().map(|(a, _)| qo(&a.recip())).collect();
                agree("recip", "nalgebra", &w, &n2.iter().map(|(a, _)| qn(&a.try_inverse().unwrap())).collect::<Vec<_>>(), tol, false);
                row!("recip", p("random quaternions"), "glam's inverse is the conjugate (unit quaternions only): not the same work", vec![
                    mk("ours", "", &o2, |(a, _)| a.recip()),
                    mk("nalgebra", "try_inverse", &n2, |(a, _)| a.try_inverse()),
                ]);

                let w: Vec<Vec<f64>> = o2.iter().map(|(a, b)| qo(&(*a / *b))).collect();
                agree("div", "nalgebra", &w, &n2.iter().map(|(a, b)| qn(&a.right_div(b).unwrap())).collect::<Vec<_>>(), tol, false);
                row!("div", p("random quaternions"), "", vec![
                    mk("ours", "a·b⁻¹", &o2, |(a, b)| *a / *b),
                    mk("nalgebra", "right_div", &n2, |(a, b)| a.right_div(b)),
                ]);

                let w: Vec<Vec<f64>> = o2.iter().map(|(a, b)| vec![a.dot(*b) as f64]).collect();
                agree("dot", "glam", &w, &g2.iter().map(|(a, b)| vec![a.dot(*b) as f64]).collect::<Vec<_>>(), tol, false);
                agree("dot", "nalgebra", &w, &n2.iter().map(|(a, b)| vec![a.dot(b) as f64]).collect::<Vec<_>>(), tol, false);
                row!("dot", p("random quaternions"), "", vec![
                    mk("ours", "", &o2, |(a, b)| a.dot(*b)),
                    mk("glam", "", &g2, |(a, b)| a.dot(*b)),
                    mk("nalgebra", "", &n2, |(a, b)| a.dot(b)),
                ]);

                // exp, ln, sqrt, powf (nalgebra only), powi/powq (none)
                let w: Vec<Vec<f64>> = o2.iter().map(|(a, _)| qo(&a.exp())).collect();
                let e = agree("exp", "nalgebra", &w, &n2.iter().map(|(a, _)| qn(&a.exp())).collect::<Vec<_>>(), tol, false);
                row!("exp", p("random quaternions"), "glam has none", vec![
                    mk("ours", "", &o2, |(a, _)| a.exp()),
                    mk("nalgebra", "", &n2, |(a, _)| a.exp()),
                ]);
                let _ = e;
                let w: Vec<Vec<f64>> = o2.iter().map(|(a, _)| qo(&a.ln())).collect();
                agree("ln", "nalgebra", &w, &n2.iter().map(|(a, _)| qn(&a.ln())).collect::<Vec<_>>(), tol, false);
                row!("ln", p("random quaternions"), "glam has none", vec![
                    mk("ours", "", &o2, |(a, _)| a.ln()),
                    mk("nalgebra", "", &n2, |(a, _)| a.ln()),
                ]);
                let w: Vec<Vec<f64>> = o2.iter().map(|(a, _)| qo(&a.sqrt())).collect();
                agree("sqrt", "nalgebra", &w, &n2.iter().map(|(a, _)| qn(&a.sqrt())).collect::<Vec<_>>(), tol * 10.0, false);
                row!("sqrt", p("random quaternions"), "glam has none", vec![
                    mk("ours", "Kahan's", &o2, |(a, _)| a.sqrt()),
                    mk("nalgebra", "", &n2, |(a, _)| a.sqrt()),
                ]);
                let op: Vec<(Q, T)> = o2.iter().zip(&ts).map(|((a, _), t)| (*a, (3.0 * t - 1.5) as T)).collect();
                let np: Vec<(NQ, T)> = n2.iter().zip(&ts).map(|((a, _), t)| (*a, (3.0 * t - 1.5) as T)).collect();
                let w: Vec<Vec<f64>> = op.iter().map(|(a, t)| qo(&a.powf(*t))).collect();
                agree("powf", "nalgebra", &w, &np.iter().map(|(a, t)| qn(&a.powf(*t))).collect::<Vec<_>>(), tol * 10.0, false);
                row!("powf", p("random quaternions, t in [-1.5, 1.5)"), "glam has none", vec![
                    mk("ours", "", &op, |(a, t)| a.powf(*t)),
                    mk("nalgebra", "", &np, |(a, t)| a.powf(*t)),
                ]);
                let oi: Vec<(Q, i32)> = o2.iter().zip(&ts).map(|((a, _), t)| (*a, (t * 20.0) as i32 - 10)).collect();
                row!("powi", p("random quaternions, n in [-10, 10)"), "no reference has it", vec![
                    mk("ours", "", &oi, |(a, n)| a.powi(*n)),
                ]);
                row!("powq", p("random quaternions"), "no reference has it", vec![
                    mk("ours", "", &o2, |(a, b)| a.powq(*b)),
                ]);
            }

            // ---- rotations --------------------------------------------------------------
            {
                let w: Vec<Vec<f64>> = ouv.iter().map(|(q, v)| vo(&q.rotate(*v))).collect();
                agree("rotate", "glam", &w, &guv.iter().map(|(q, v)| vg(&(*q * *v))).collect::<Vec<_>>(), tol, false);
                agree("rotate", "nalgebra", &w, &nuv.iter().map(|(q, v)| vn(&(q * v))).collect::<Vec<_>>(), tol, false);
                row!("rotate", p("unit quaternions"), "glam and nalgebra assume |q| = 1; ours divides by |q|²", vec![
                    mk("ours", "", &ouv, |(q, v)| q.rotate(*v)),
                    mk("glam", "q * v", &guv, |(q, v)| *q * *v),
                    mk("nalgebra", "UnitQuaternion * Vector3", &nuv, |(q, v)| q * v),
                ]);
                let w: Vec<Vec<f64>> = ouv.iter().map(|(q, v)| vo(&q.rotate_inverse(*v))).collect();
                agree("rotinv", "glam", &w, &guv.iter().map(|(q, v)| vg(&(q.conjugate() * *v))).collect::<Vec<_>>(), tol, false);
                agree("rotinv", "nalgebra", &w, &nuv.iter().map(|(q, v)| vn(&q.inverse_transform_vector(v))).collect::<Vec<_>>(), tol, false);
                row!("rotate_inverse", p("unit quaternions"), "", vec![
                    mk("ours", "", &ouv, |(q, v)| q.rotate_inverse(*v)),
                    mk("glam", "q.conjugate() * v", &guv, |(q, v)| q.conjugate() * *v),
                    mk("nalgebra", "inverse_transform_vector", &nuv, |(q, v)| q.inverse_transform_vector(v)),
                ]);

                // slerp and nlerp
                let w: Vec<Vec<f64>> = ou.iter().map(|(a, b, t)| qo(&a.slerp(*b, *t))).collect();
                agree("slerp", "glam", &w, &gu.iter().map(|(a, b, t)| qg(&a.slerp(*b, *t))).collect::<Vec<_>>(), tol * 100.0, true);
                agree("slerp", "nalgebra", &w, &nuu.iter().map(|(a, b, t)| qn(&a.slerp(b, *t).into_inner())).collect::<Vec<_>>(), tol * 100.0, true);
                row!("slerp", p("unit quaternions, t in [0, 1)"), "ours normalises both first; the others assume unit inputs", vec![
                    mk("ours", "", &ou, |(a, b, t)| a.slerp(*b, *t)),
                    mk("glam", "", &gu, |(a, b, t)| a.slerp(*b, *t)),
                    mk("nalgebra", "UnitQuaternion::slerp", &nuu, |(a, b, t)| a.slerp(b, *t)),
                ]);
                let w: Vec<Vec<f64>> = ou.iter().map(|(a, b, t)| qo(&a.nlerp(*b, *t))).collect();
                agree("nlerp", "glam", &w, &gu.iter().map(|(a, b, t)| qg(&a.lerp(*b, *t))).collect::<Vec<_>>(), tol * 10.0, true);
                let nl_ok = nuu
                    .iter()
                    .zip(&w)
                    .all(|((a, b, t), w)| agree("nlerp", "nalgebra", &[w.clone()], &[qn(&a.nlerp(b, *t).into_inner())], 1.0, true) < tol * 10.0);
                row!("nlerp", p("unit quaternions, t in [0, 1)"), "", vec![
                    mk("ours", "", &ou, |(a, b, t)| a.nlerp(*b, *t)),
                    mk("glam", "lerp (normalised, shorter arc)", &gu, |(a, b, t)| a.lerp(*b, *t)),
                    mk("nalgebra", if nl_ok { "UnitQuaternion::nlerp" } else { "UnitQuaternion::nlerp: not on the shorter arc, differs from ours on part of the pool" }, &nuu, |(a, b, t)| a.nlerp(b, *t)),
                ]);

                // angles between and of
                let w: Vec<Vec<f64>> = ou.iter().map(|(a, b, _)| vec![a.angle_to(*b) as f64]).collect();
                agree("angle_to", "glam", &w, &gu.iter().map(|(a, b, _)| vec![a.angle_between(*b) as f64]).collect::<Vec<_>>(), tol * 1e3, false);
                agree("angle_to", "nalgebra", &w, &nuu.iter().map(|(a, b, _)| vec![a.angle_to(b) as f64]).collect::<Vec<_>>(), tol * 1e3, false);
                row!("angle_to", p("unit quaternions"), "ours by atan2 (accurate near 0 and π); glam by acos of the dot product", vec![
                    mk("ours", "", &ou, |(a, b, _)| a.angle_to(*b)),
                    mk("glam", "angle_between", &gu, |(a, b, _)| a.angle_between(*b)),
                    mk("nalgebra", "", &nuu, |(a, b, _)| a.angle_to(b)),
                ]);
                let w: Vec<Vec<f64>> = ou.iter().map(|(a, _, _)| vec![a.angle() as f64]).collect();
                agree("angle", "nalgebra", &w, &nuu.iter().map(|(a, _, _)| vec![a.angle() as f64]).collect::<Vec<_>>(), tol * 1e3, false);
                row!("angle", p("unit quaternions"), "", vec![
                    mk("ours", "", &ou, |(a, _, _)| a.angle()),
                    mk("glam", "to_axis_angle().1: also forms the axis; angle in [0, 2π)", &gu, |(a, _, _)| a.to_axis_angle().1),
                    mk("nalgebra", "UnitQuaternion::angle", &nuu, |(a, _, _)| a.angle()),
                ]);

                // matrices
                let w: Vec<Vec<f64>> = ou.iter().map(|(a, _, _)| a.to_rotation_matrix().iter().flatten().map(|x| *x as f64).collect()).collect();
                let gm = |m: &GM| -> Vec<f64> { (0..3).flat_map(|r| (0..3).map(move |c| m.col(c)[r] as f64)).collect() };
                agree("to_rotation_matrix", "glam", &w, &gu.iter().map(|(a, _, _)| gm(&GM::from_quat(*a))).collect::<Vec<_>>(), tol, false);
                let nm = |m: &nalgebra::Rotation3<T>| -> Vec<f64> { (0..3).flat_map(|r| (0..3).map(move |c| m[(r, c)] as f64)).collect() };
                agree("to_rotation_matrix", "nalgebra", &w, &nuu.iter().map(|(a, _, _)| nm(&a.to_rotation_matrix())).collect::<Vec<_>>(), tol, false);
                row!("to_rotation_matrix", p("unit quaternions"), "ours divides by |q|²", vec![
                    mk("ours", "[[T; 3]; 3]", &ou, |(a, _, _)| a.to_rotation_matrix()),
                    mk("glam", "Mat3::from_quat", &gu, |(a, _, _)| GM::from_quat(*a)),
                    mk("nalgebra", "to_rotation_matrix", &nuu, |(a, _, _)| a.to_rotation_matrix()),
                ]);
                row!("to_matrix", p("unit quaternions"), "ours into a heap-allocated a::math::matrix::Matrix", vec![
                    mk("ours", "Matrix<T>", &ou, |(a, _, _)| a.to_matrix()),
                    mk("glam", "Mat3::from_quat", &gu, |(a, _, _)| GM::from_quat(*a)),
                    mk("nalgebra", "to_rotation_matrix", &nuu, |(a, _, _)| a.to_rotation_matrix()),
                ]);
                let om: Vec<([[T; 3]; 3], ())> = ou.iter().map(|(a, _, _)| (a.to_rotation_matrix(), ())).collect();
                let omm: Vec<(Matrix<T>, ())> = ou.iter().map(|(a, _, _)| (a.to_matrix(), ())).collect();
                let gmm: Vec<(GM, ())> = om.iter().map(|(m, _)| (GM::from_cols_array_2d(&[[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]), ())).collect();
                let nmm: Vec<(nalgebra::Rotation3<T>, ())> = om
                    .iter()
                    .map(|(m, _)| (nalgebra::Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2])), ()))
                    .collect();
                let w: Vec<Vec<f64>> = om.iter().map(|(m, _)| qo(&Q::from_rotation_matrix(*m))).collect();
                agree("from_rotation_matrix", "glam", &w, &gmm.iter().map(|(m, _)| qg(&GQ::from_mat3(m))).collect::<Vec<_>>(), tol * 10.0, true);
                agree("from_rotation_matrix", "nalgebra", &w, &nmm.iter().map(|(m, _)| qn(&NU::from_rotation_matrix(m).into_inner())).collect::<Vec<_>>(), tol * 10.0, true);
                row!("from_rotation_matrix", p("rotation matrices"), "ours normalises the result (Shepperd); glam assumes a pure rotation", vec![
                    mk("ours", "", &om, |(m, _)| Q::from_rotation_matrix(*m)),
                    mk("glam", "Quat::from_mat3", &gmm, |(m, _)| GQ::from_mat3(m)),
                    mk("nalgebra", "UnitQuaternion::from_rotation_matrix", &nmm, |(m, _)| NU::from_rotation_matrix(m)),
                ]);
                row!("from_matrix", p("rotation matrices"), "ours from an a::math::matrix::Matrix", vec![
                    mk("ours", "&Matrix<T>", &omm, |(m, _)| Q::from_matrix(m)),
                    mk("glam", "Quat::from_mat3", &gmm, |(m, _)| GQ::from_mat3(m)),
                    mk("nalgebra", "UnitQuaternion::from_rotation_matrix", &nmm, |(m, _)| NU::from_rotation_matrix(m)),
                ]);

                // axis and angle, rotation vectors, arcs
                let aa: Vec<([f64; 3], f64)> = vs
                    .iter()
                    .zip(&ts)
                    .map(|(v, t)| {
                        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                        ([v[0] / n, v[1] / n, v[2] / n], 6.0 * t - 3.0)
                    })
                    .collect();
                let oaa: Vec<([T; 3], T)> = aa.iter().map(|(v, t)| (ov(v), *t as T)).collect();
                let gaa: Vec<(GV, T)> = aa.iter().map(|(v, t)| (gv(v), *t as T)).collect();
                let naa: Vec<(nalgebra::Unit<NV>, T)> = aa.iter().map(|(v, t)| (nalgebra::Unit::new_unchecked(nv(v)), *t as T)).collect();
                let w: Vec<Vec<f64>> = oaa.iter().map(|(v, t)| qo(&Q::from_axis_angle(*v, *t))).collect();
                agree("from_axis_angle", "glam", &w, &gaa.iter().map(|(v, t)| qg(&GQ::from_axis_angle(*v, *t))).collect::<Vec<_>>(), tol, false);
                agree("from_axis_angle", "nalgebra", &w, &naa.iter().map(|(v, t)| qn(&NU::from_axis_angle(v, *t).into_inner())).collect::<Vec<_>>(), tol, false);
                row!("from_axis_angle", p("unit axes"), "ours normalises the axis; the others take it as unit", vec![
                    mk("ours", "", &oaa, |(v, t)| Q::from_axis_angle(*v, *t)),
                    mk("glam", "", &gaa, |(v, t)| GQ::from_axis_angle(*v, *t)),
                    mk("nalgebra", "", &naa, |(v, t)| NU::from_axis_angle(v, *t)),
                ]);
                // compared as the rotations they give (glam's angle is in [0, 2π), ours in [0, π])
                let w: Vec<Vec<f64>> = ou.iter().map(|(a, _, _)| { let (ax, an) = a.to_axis_angle(); qo(&Q::from_axis_angle(ax, an)) }).collect();
                agree("to_axis_angle", "glam", &w, &gu.iter().map(|(a, _, _)| { let (ax, an) = a.to_axis_angle(); qo(&Q::from_axis_angle([ax.x, ax.y, ax.z], an)) }).collect::<Vec<_>>(), tol * 1e3, true);
                agree("to_axis_angle", "nalgebra", &w, &nuu.iter().map(|(a, _, _)| { let (ax, an) = a.axis_angle().unwrap(); qo(&Q::from_axis_angle([ax.x, ax.y, ax.z], an)) }).collect::<Vec<_>>(), tol * 1e3, true);
                row!("to_axis_angle", p("unit quaternions"), "", vec![
                    mk("ours", "", &ou, |(a, _, _)| a.to_axis_angle()),
                    mk("glam", "", &gu, |(a, _, _)| a.to_axis_angle()),
                    mk("nalgebra", "axis_angle (Option)", &nuu, |(a, _, _)| a.axis_angle()),
                ]);
                let orv: Vec<([T; 3], ())> = vs.iter().map(|v| (ov(v), ())).collect();
                let grv: Vec<(GV, ())> = vs.iter().map(|v| (gv(v), ())).collect();
                let nrv: Vec<(NV, ())> = vs.iter().map(|v| (nv(v), ())).collect();
                let w: Vec<Vec<f64>> = orv.iter().map(|(v, _)| qo(&Q::from_rotation_vector(*v))).collect();
                agree("from_rotation_vector", "glam", &w, &grv.iter().map(|(v, _)| qg(&GQ::from_scaled_axis(*v))).collect::<Vec<_>>(), tol, false);
                agree("from_rotation_vector", "nalgebra", &w, &nrv.iter().map(|(v, _)| qn(&NU::from_scaled_axis(*v).into_inner())).collect::<Vec<_>>(), tol, false);
                row!("from_rotation_vector", p("random vectors"), "", vec![
                    mk("ours", "", &orv, |(v, _)| Q::from_rotation_vector(*v)),
                    mk("glam", "from_scaled_axis", &grv, |(v, _)| GQ::from_scaled_axis(*v)),
                    mk("nalgebra", "from_scaled_axis", &nrv, |(v, _)| NU::from_scaled_axis(*v)),
                ]);
                let w: Vec<Vec<f64>> = ou.iter().map(|(a, _, _)| vo(&a.to_rotation_vector())).collect();
                let wq: Vec<Vec<f64>> = ou.iter().map(|(a, _, _)| qo(&Q::from_rotation_vector(a.to_rotation_vector()))).collect();
                agree("to_rotation_vector", "nalgebra", &wq, &nuu.iter().map(|(a, _, _)| { let v = a.scaled_axis(); qo(&Q::from_rotation_vector([v.x, v.y, v.z])) }).collect::<Vec<_>>(), tol * 1e3, true);
                agree("to_rotation_vector", "glam", &wq, &gu.iter().map(|(a, _, _)| { let v = a.to_scaled_axis(); qo(&Q::from_rotation_vector([v.x, v.y, v.z])) }).collect::<Vec<_>>(), tol * 1e3, true);
                let g_ok = gu.iter().zip(&w).all(|((a, _, _), w)| agree("trv", "glam", &[w.clone()], &[vg(&a.to_scaled_axis())], 1e9, false) < tol * 1e3);
                row!("to_rotation_vector", p("unit quaternions"), "", vec![
                    mk("ours", "angle in [0, π]", &ou, |(a, _, _)| a.to_rotation_vector()),
                    mk("glam", if g_ok { "to_scaled_axis" } else { "to_scaled_axis: angle in [0, 2π), differs from ours where w < 0" }, &gu, |(a, _, _)| a.to_scaled_axis()),
                    mk("nalgebra", "scaled_axis", &nuu, |(a, _, _)| a.scaled_axis()),
                ]);
                let arc: Vec<([f64; 3], [f64; 3])> = (0..len)
                    .map(|_| {
                        let u = |rng: &mut Rng| {
                            let v = rv(rng);
                            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                            [v[0] / n, v[1] / n, v[2] / n]
                        };
                        (u(&mut rng), u(&mut rng))
                    })
                    .collect();
                let oa: Vec<([T; 3], [T; 3])> = arc.iter().map(|(a, b)| (ov(a), ov(b))).collect();
                let ga: Vec<(GV, GV)> = arc.iter().map(|(a, b)| (gv(a), gv(b))).collect();
                let na: Vec<(NV, NV)> = arc.iter().map(|(a, b)| (nv(a), nv(b))).collect();
                let w: Vec<Vec<f64>> = oa.iter().map(|(a, b)| qo(&Q::from_rotation_arc(*a, *b))).collect();
                agree("from_rotation_arc", "glam", &w, &ga.iter().map(|(a, b)| qg(&GQ::from_rotation_arc(*a, *b))).collect::<Vec<_>>(), tol * 100.0, true);
                agree("from_rotation_arc", "nalgebra", &w, &na.iter().map(|(a, b)| qn(&NU::rotation_between(a, b).unwrap().into_inner())).collect::<Vec<_>>(), tol * 100.0, true);
                row!("from_rotation_arc", p("unit vectors"), "ours takes any lengths (normalises); glam requires unit vectors", vec![
                    mk("ours", "", &oa, |(a, b)| Q::from_rotation_arc(*a, *b)),
                    mk("glam", "", &ga, |(a, b)| GQ::from_rotation_arc(*a, *b)),
                    mk("nalgebra", "rotation_between (Option)", &na, |(a, b)| NU::rotation_between(a, b)),
                ]);

                // Euler angles: three angles in ranges away from gimbal lock
                let eu: Vec<[f64; 3]> = (0..len).map(|_| [rng.range(-3.0, 3.0), rng.range(-1.4, 1.4), rng.range(-3.0, 3.0)]).collect();
                let eu_p: Vec<[f64; 3]> = (0..len).map(|_| [rng.range(-3.0, 3.0), rng.range(0.2, 2.9), rng.range(-3.0, 3.0)]).collect();
                for (seq, frame, gr, label, proper) in [
                    (EulerSequence::XYZ, EulerFrame::Intrinsic, EulerRot::XYZ, "XYZ intrinsic", false),
                    (EulerSequence::ZYX, EulerFrame::Intrinsic, EulerRot::ZYX, "ZYX intrinsic (yaw, pitch, roll)", false),
                    (EulerSequence::XYZ, EulerFrame::Extrinsic, EulerRot::XYZEx, "XYZ extrinsic (roll, pitch, yaw)", false),
                    (EulerSequence::ZXZ, EulerFrame::Intrinsic, EulerRot::ZXZ, "ZXZ intrinsic (proper)", true),
                ] {
                    let src = if proper { &eu_p } else { &eu };
                    let oe: Vec<([T; 3], ())> = src.iter().map(|e| (ov(e), ())).collect();
                    let w: Vec<Vec<f64>> = oe.iter().map(|(e, _)| qo(&Q::from_euler(seq, frame, *e))).collect();
                    agree("from_euler", "glam", &w, &oe.iter().map(|(e, _)| qg(&GQ::from_euler(gr, e[0], e[1], e[2]))).collect::<Vec<_>>(), tol, true);
                    let mut s = vec![
                        mk("ours", "", &oe, move |(e, _)| Q::from_euler(seq, frame, *e)),
                        mk("glam", "", &oe, move |(e, _)| GQ::from_euler(gr, e[0], e[1], e[2])),
                    ];
                    let nal = seq == EulerSequence::XYZ && frame == EulerFrame::Extrinsic;
                    if nal {
                        agree("from_euler", "nalgebra", &w, &oe.iter().map(|(e, _)| qn(&NU::from_euler_angles(e[0], e[1], e[2]).into_inner())).collect::<Vec<_>>(), tol, true);
                        s.push(mk("nalgebra", "from_euler_angles", &oe, |(e, _)| NU::from_euler_angles(e[0], e[1], e[2])));
                    }
                    row!("from_euler", p(label), "", s);

                    let oq: Vec<(Q, ())> = oe.iter().map(|(e, _)| (Q::from_euler(seq, frame, *e), ())).collect();
                    let gqv: Vec<(GQ, ())> = oe.iter().map(|(e, _)| (GQ::from_euler(gr, e[0], e[1], e[2]), ())).collect();
                    // compared as the rotations the angles give: a proper sequence has two triples
                    let back = |e: [T; 3]| qo(&Q::from_euler(seq, frame, e));
                    let w: Vec<Vec<f64>> = oq.iter().map(|(q, _)| back(q.to_euler(seq, frame))).collect();
                    agree("to_euler", "glam", &w, &gqv.iter().map(|(q, _)| { let (a, b, c) = q.to_euler(gr); back([a, b, c]) }).collect::<Vec<_>>(), tol * 1e3, true);
                    let same_triple = oq.iter().zip(&gqv).all(|((q, _), (g, _))| {
                        let (a, b, c) = g.to_euler(gr);
                        let e = q.to_euler(seq, frame);
                        ((e[0] - a).abs() + (e[1] - b).abs() + (e[2] - c).abs()) < 1e-3
                    });
                    let mut s = vec![
                        mk("ours", "Bernardes and Viollet, any sequence", &oq, move |(q, _)| q.to_euler(seq, frame)),
                        mk("glam", if same_triple { "" } else { "the other triple of the same rotation (middle angle negative)" }, &gqv, move |(q, _)| q.to_euler(gr)),
                    ];
                    if nal {
                        let nqv: Vec<(NU, ())> = oe.iter().map(|(e, _)| (NU::from_euler_angles(e[0], e[1], e[2]), ())).collect();
                        agree("to_euler", "nalgebra", &w, &nqv.iter().map(|(q, _)| { let (a, b, c) = q.euler_angles(); back([a, b, c]) }).collect::<Vec<_>>(), tol * 1e3, true);
                        let nqv: &'static [(NU, ())] = Box::leak(nqv.into_boxed_slice());
                        s.push(mk("nalgebra", "euler_angles (through the rotation matrix)", nqv, |(q, _)| q.euler_angles()));
                    }
                    row!("to_euler", p(label), "", s);
                }
            }
        }
    };
}

quat_bench!(run_f32, f32, "f32", glam::Quat, glam::Vec3, glam::Mat3, 2e-5);
quat_bench!(run_f64, f64, "f64", glam::DQuat, glam::DVec3, glam::DMat3, 1e-12);

fn main() {
    let out = Out::new("numtypes");
    run_f32(&out, 7);
    run_f64(&out, 8);
}
