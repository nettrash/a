mod ffi;
#[macro_use]
mod h;
mod g_arith;
mod g_func;
mod g_conv;
mod g_const;
mod g_cplx;
mod g_mat;
mod libs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    unsafe {
        ffi::mpfr_set_emax(ffi::mpfr_get_emax_max());
        ffi::mpfr_set_emin(ffi::mpfr_get_emin_min());
    }
    let mut it = args.iter().skip(1);
    let mut group = String::new();
    let mut sub = String::new();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--filter" => unsafe { (*&raw mut h::CFG).filter.push(it.next().unwrap().clone()) },
            "--quick" => unsafe { h::CFG.quick = true },
            "--check" => unsafe { h::CFG.check_only = true },
            "--skip" => unsafe { (*&raw mut h::CFG).skip.push(it.next().unwrap().clone()) },
            "--samples" => unsafe { h::CFG.samples = it.next().unwrap().parse().unwrap() },
            _ if group.is_empty() => group = a.clone(),
            _ => sub = a.clone(),
        }
    }
    match group.as_str() {
        "arith" => g_arith::run(&sub),
        "func" => g_func::run(&sub),
        "conv" => g_conv::run("conv"),
        "text" => g_conv::run("text"),
        "consts" => g_const::run(&sub),
        "complex" => g_cplx::run(&sub),
        "matrix" => g_mat::run(&sub),
        "once" => {
            let a: Vec<&String> = args.iter().skip(2).collect();
            g_const::once(a[0], a[1], a[2].parse().unwrap());
        }
        "info" => {
            let v = unsafe { std::ffi::CStr::from_ptr(ffi::mpfr_get_version()) };
            let c = unsafe { std::ffi::CStr::from_ptr(ffi::mpc_get_version()) };
            println!("MPFR {} MPC {} libs {}", v.to_string_lossy(), c.to_string_lossy(), env!("FB_LIBS"));
        }
        _ => panic!("usage: floatbench <group> <sub> [--filter s] [--quick] [--samples n]"),
    }
}
