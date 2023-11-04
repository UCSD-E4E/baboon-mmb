mod amfd;
mod lrmc;
mod pf;

#[derive(structopt::StructOpt, Debug)]
#[structopt(name = "MMB Paper parameters")]
pub struct Opt {
    #[structopt(long, default_value = "4")]
    k: f64,

    #[structopt(long, possible_values = &["4", "8"], default_value = "8")]
    connectivity: i32,

    #[structopt(long, default_value = "5")]
    area_min: i32,

    #[structopt(long, default_value = "80")]
    area_max: i32,

    #[structopt(long, default_value = "1.0")]
    aspect_ratio_min: f64,

    #[structopt(long, default_value = "6.0")]
    aspect_ratio_max: f64,

    #[structopt(long, default_value = "4")]
    l: i32,

    #[structopt(long, possible_values = &["1", "3", "5", "7", "9", "11"], default_value = "3")]
    kernel: i32,

    #[structopt(long, default_value = "5")]
    pipeline_length: i32,

    #[structopt(long, possible_values = &["3", "5", "7", "9", "11"], default_value = "7")]
    pipeline_size: i32,

    #[structopt(long, default_value = "3")]
    h: i32,

    #[structopt(long, default_value = "10")]
    max_niter_param: i32,

    #[structopt(long, default_value = "0.8")]
    gamma1_param: f64,

    #[structopt(long, default_value = "0.8")]
    gamma2_param: f64,

    video_file: String,
}

fn main() {
    let opt = <Opt as structopt::StructOpt>::from_args();

    if !std::path::Path::new("processing/amfd").exists() {
        std::fs::create_dir_all("processing/amfd").unwrap();
    }
    if !std::path::Path::new("processing/lrmc").exists() {
        std::fs::create_dir_all("processing/lrmc").unwrap();
    }
    if !std::path::Path::new("processing/frames").exists() {
        std::fs::create_dir_all("processing/frames").unwrap();
    }

    let start_amfd = std::time::Instant::now();
    amfd::amfd(&opt);
    let duration_amfd: std::time::Duration = start_amfd.elapsed();
    println!("AMFD done in {:?}", duration_amfd);

    let start_pf = std::time::Instant::now();
    lrmc::lrmc(&opt);
    let duration_pf: std::time::Duration = start_pf.elapsed();
    println!("LRMC done in {:?}", duration_pf);

    let start_lrmc = std::time::Instant::now();
    pf::pf(&opt);
    let duration_lrmc: std::time::Duration = start_lrmc.elapsed();
    println!("PF done in {:?}", duration_lrmc);

    std::fs::remove_dir_all("processing").unwrap();
    println!("Done!");
}
