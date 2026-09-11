use rang_toy::{Config, run};

fn main() {
    if let Err(error) = entry() {
        eprintln!("rang-toy: {error}");
        std::process::exit(2);
    }
}

fn entry() -> Result<(), String> {
    let mut config = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--help" {
            println!(
                "rang-toy [--seed INTEGER] [--noise JY] [--pointing ARCMIN] [--beam-error FRACTION] [--budget FRACTION] [--target-flux JY]\nPrint a JSON result to stdout. No files are written."
            );
            return Ok(());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if key == "--seed" {
            config.seed = value
                .parse()
                .map_err(|_| "seed must be an unsigned integer")?;
            continue;
        }
        let value: f64 = value
            .parse()
            .map_err(|_| format!("invalid number for {key}"))?;
        match key.as_str() {
            "--noise" => config.noise = value,
            "--pointing" => config.pointing = value,
            "--beam-error" => config.beam_error = value,
            "--budget" => config.budget = value,
            "--target-flux" => config.target_flux = value,
            _ => return Err(format!("unknown option {key}")),
        }
    }
    println!("{}", run(config)?.json());
    Ok(())
}
