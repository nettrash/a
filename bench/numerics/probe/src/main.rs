//! The numerics scoreboard probe: `numprobe <group>...` prints TSV rows (no header).
mod argmin_probe;
mod harness;
mod roots;
mod systems;
mod quad;
mod gauss;
mod ode;
mod opt;
mod interp;
mod pyside;
mod nist;

unsafe extern "C" {
    fn probe_gsl_init();
}

fn main() {
    unsafe { probe_gsl_init() };
    let args: Vec<String> = std::env::args().skip(1).collect();
    for g in &args {
        match g.as_str() {
            "roots" => roots::run(),
            "systems" => systems::run(),
            "quad" => quad::run(),
            "gauss" => gauss::run(),
            "ode" => ode::run(),
            "opt" => opt::run(),
            "interp" => interp::run(),
            "lp" => pyside::lp(),
            "lbfgsb" => pyside::lbfgsb(),
            "cubature" => pyside::cub(),
            "aaa" => pyside::aaa_group(),
            "export" => {
                pyside::export(&std::env::var("NUMPROBE_EXPORT").expect("NUMPROBE_EXPORT"));
            }
            other => eprintln!("unknown group {other}"),
        }
    }
}
