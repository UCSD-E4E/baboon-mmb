// ============================================================================
// This is a Rust implementation of the paper:
// Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark
// Qian Yin∗, QingyongHu∗, HaoLiu, Feng Zhang, Yingqian Wang, Zaiping Lin, Wei An, Yulan Guo
// This is NOT the original implemtation by the authors, as there seems to be no code available.
// ============================================================================

mod amfd;
mod lrmc;
mod pf;

// ============================================================================
// List of parameters with their default values, suggested range, and reference.
// ============================================================================
// k: 4, [0.0, 8.0], Eq. 4
//
// connectivity: 8, {4, 8}, Algorithm 1
//
// area_min: 5, [0, area_max), Eq. 7
//
// area_max: 80, (area_min, 100], Eq. 7
//
// aspect_ratio_min: 1.0, [0.0, aspect_ratio_max), Eq. 7
//
// aspect_ratio_max: 6.0, (aspect_ratio_min, 10.0], Eq. 7
//
// l: 4, [1, 10], Eq. 9
//
// kernel: 3, {1, 3, 5, 7, 9, 11}, Algorithm 1
//
// pipeline_length: 5, [1, 10], Step 1 of Pipeline Filter
//
// pipeline_size: 7, {3, 5, 7, 9, 11}, Step 1 of Pipeline Filter
//
// h: 3, [1, pipeline_length], Step 4 of Pipeline Filter
//
// max_niter_param: 10, [1, 20], fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017)
//
// gamma1_param: 0.8, [0.0, 1.0], fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017)
//
// gamma2_param: 0.8, [gamma1_param, 1.0], fRMC (Background subtraction via fast robust matrix completion, Rezaei et al., 2017)
//
// video_file: String, path to video file
// ============================================================================
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

    // Rather than storing the resulting binary masks from AMFD and LRMC in memory, we opt to save them to disk. 
    // Although it may be feasible to merge the AMFD and LRMC processes (and possibly the Pipeline Filter process), we decided to keep them separate. 
    // Combining them could make the code harder to understand and maintain, and could also result in a significant increase in the overhead required by the MATLAB scripts. 
    // Additionally, we save the frames to disk since we only need to retrieve them again for the Pipeline Filter step. 
    // By doing so, we avoid keeping them in memory when they are not needed for an extended period.
    if !std::path::Path::new("processing/amfd").exists() {
        std::fs::create_dir_all("processing/amfd").unwrap();
    }
    if !std::path::Path::new("processing/lrmc").exists() {
        std::fs::create_dir_all("processing/lrmc").unwrap();
    }

    // The LRMC algorithm necessitates the division of the video into individual frames, as certain operating systems pose difficulties for MATLAB's video reading capabilities.
    // However, relying solely on frames is insufficient, as knowledge of the video's frame rate is also needed.
    if !std::path::Path::new("processing/frames").exists() { 
        std::fs::create_dir_all("processing/frames").unwrap();
    }


    // Start and montior the Accumulative Multi-Frame Difference (AMFD) process.
    let start_amfd = std::time::Instant::now();
    amfd::amfd(&opt);
    let duration_amfd: std::time::Duration = start_amfd.elapsed();
    println!("AMFD done in {:?}", duration_amfd);

    // Start and montior the Low-Rank Matrix Completion (LRMC) process.
    let start_lrmc = std::time::Instant::now();
    lrmc::lrmc(&opt);
    let duration_lrmc: std::time::Duration = start_lrmc.elapsed();
    println!("LRMC done in {:?}", duration_lrmc);

    // Start and montior the Pipeline Filter (PF) process.
    let start_pf = std::time::Instant::now();
    pf::pf(&opt);
    let duration_pf: std::time::Duration = start_pf.elapsed();
    println!("PF done in {:?}", duration_pf);

    // Remove the temporary directories and their contents.
    std::fs::remove_dir_all("processing").unwrap();
    println!("Done!");
}
